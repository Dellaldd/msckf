/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <boost/math/distributions/chi_squared.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>


#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <msckf_vio/msckf_vio.h>
#include <msckf_vio/math_utils.hpp>
#include <msckf_vio/utils.h>
#include <ceres/ceres.h>
#include <msckf_vio/BiasError.h>
#include <msckf_vio/InitializeOpti.h> 


using namespace std;
using namespace Eigen;

namespace msckf_vio{
// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();

// Static member variables in CAMState class.
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> MsckfVio::chi_squared_test_table;

MsckfVio::MsckfVio(ros::NodeHandle& pnh):
  is_gravity_set(false),
  is_first_img(true),
  nh(pnh) {
  return;
}

bool MsckfVio::loadParameters() {
  state_server.imu_state.gyro_bias = Eigen::Vector3d(-0.002341, 0.021815, 0.076602);
  // state_server.imu_state.gyro_bias = Eigen::Vector3d(-0.002153, 0.020744, 0.075806);
  // state_server.imu_state.gyro_bias = Eigen::Vector3d(-0.0546303, 0.0208792, 0.094797);


  // gt
  ifstream ifs_gt;
  

  nh.param<string>("gt_path", gt_path, "/home/ldd/euroc/V1_01_easy/mav0/state_groundtruth_estimate0/V1_01_easy.txt");
  nh.param<string>("gt_type", gt_type, "euroc");
  nh.param<bool>("use_gt_initial", use_gt_initial, true);
  

  // gravity
  if(gt_type == "real"){
    IMUState::gravity = Vector3d(0, 0, -9.81);
  }else{
    IMUState::gravity = Vector3d(0, 0, -9.81);
  }
  
  ifs_gt.open(gt_path, ios::in);  //读取文件
  if(!ifs_gt.is_open())
  {
      std::cout << "gt ifstream open file error!\n";
  }
  string line_gt;
  vector<string> lines_gt;
  while(getline(ifs_gt, line_gt))    
      lines_gt.push_back(line_gt);   
  ifs_gt.close();
  Gt gt_pose;
  string s_gt;
  vector<std::string> vec_gt;

  // euroc 
  if(gt_type == "euroc"){
    cout << "gt_type: " << gt_type << endl;
    s_gt = ",";
    for(int i=1; i<lines_gt.size(); ++i){
      vec_gt = split_vec(lines_gt[i],s_gt);
      gt_pose.time = std::stold(vec_gt[0]) * 1e-9;
      gt_pose.p << std::stod(vec_gt[1]), std::stod(vec_gt[2]), std::stod(vec_gt[3]);
      gt_pose.q.x() = std::stod(vec_gt[5]);
      gt_pose.q.y() = std::stod(vec_gt[6]);
      gt_pose.q.z() = std::stod(vec_gt[7]);
      gt_pose.q.w() = std::stod(vec_gt[4]);
      gt_pose.vel = Eigen::Vector3d(std::stod(vec_gt[8]), std::stod(vec_gt[9]), std::stod(vec_gt[10]));
      gt_poses.push_back(gt_pose);
    }
  }

  if(gt_type == "euroc_gt"){
    s_gt = " ";
    for(int i=1; i<lines_gt.size(); ++i){
      vec_gt = split_vec(lines_gt[i],s_gt);
      gt_pose.time = std::stold(vec_gt[0]);
      gt_pose.p << std::stod(vec_gt[1]), std::stod(vec_gt[2]), std::stod(vec_gt[3]);
      gt_pose.q.x() = std::stod(vec_gt[5]);
      gt_pose.q.y() = std::stod(vec_gt[6]);
      gt_pose.q.z() = std::stod(vec_gt[7]);
      gt_pose.q.w() = std::stod(vec_gt[4]);
      gt_pose.vel = Eigen::Vector3d(std::stod(vec_gt[8]), std::stod(vec_gt[9]), std::stod(vec_gt[10]));
      gt_poses.push_back(gt_pose);
    }
  }
    

    // euroc v1
    if(gt_type == "eurocv1" || gt_type == "real"){
      s_gt = " ";
      for(int i=1; i<lines_gt.size(); ++i){
          vec_gt = split_vec(lines_gt[i],s_gt);
          gt_pose.time = std::stold(vec_gt[0]);
          gt_pose.p << std::stod(vec_gt[1]), std::stod(vec_gt[2]), std::stod(vec_gt[3]);
          gt_pose.q.x() = std::stod(vec_gt[4]);
          gt_pose.q.y() = std::stod(vec_gt[5]);
          gt_pose.q.z() = std::stod(vec_gt[6]);
          gt_pose.q.w() = std::stod(vec_gt[7]);
          gt_poses.push_back(gt_pose);
        }
    }

    // simulation dataset
    if(gt_type == "simulation"){
      s_gt = " ";
      for(int i=1; i<lines_gt.size(); ++i){
          vec_gt = split_vec(lines_gt[i],s_gt);
          gt_pose.time = std::stold(vec_gt[0]);
          gt_pose.p << std::stod(vec_gt[5]), std::stod(vec_gt[6]), std::stod(vec_gt[7]);

          gt_pose.q.x() = std::stod(vec_gt[2]);
          gt_pose.q.y() = std::stod(vec_gt[3]);
          gt_pose.q.z() = std::stod(vec_gt[4]);
          gt_pose.q.w() = std::stod(vec_gt[1]);
          gt_pose.vel = Vector3d(std::stod(vec_gt[8]), std::stod(vec_gt[9]), std::stod(vec_gt[10]));
          gt_poses.push_back(gt_pose);
      }
    }

  cout << "GT_SIZE: " << gt_poses.size() << endl;
  // Frame id
  nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
  nh.param<string>("child_frame_id", child_frame_id, "robot");
  nh.param<bool>("publish_tf", publish_tf, true);
  nh.param<double>("frame_rate", frame_rate, 40.0);
  nh.param<double>("position_std_threshold", position_std_threshold, 8.0);

  nh.param<double>("rotation_threshold", rotation_threshold, 0.2618);
  nh.param<double>("translation_threshold", translation_threshold, 0.4);
  nh.param<double>("tracking_rate_threshold", tracking_rate_threshold, 0.5);

  // Feature optimization parameters
  nh.param<double>("feature/config/translation_threshold",
      Feature::optimization_config.translation_threshold, 0.2);

  // Noise related parameters
  nh.param<double>("noise/gyro", IMUState::gyro_noise, 0.001);
  nh.param<double>("noise/acc", IMUState::acc_noise, 0.01);
  nh.param<double>("noise/gyro_bias", IMUState::gyro_bias_noise, 0.001);
  nh.param<double>("noise/acc_bias", IMUState::acc_bias_noise, 0.01);
  nh.param<double>("noise/feature", Feature::observation_noise, 0.01);

  // Use variance instead of standard deviation.
  IMUState::gyro_noise *= IMUState::gyro_noise;
  IMUState::acc_noise *= IMUState::acc_noise;
  IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
  IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
  Feature::observation_noise *= Feature::observation_noise;

  // Set the initial IMU state.
  // The intial orientation and position will be set to the origin
  // implicitly. But the initial velocity and bias can be
  // set by parameters.
  // TODO: is it reasonable to set the initial bias to 0?
  nh.param<double>("initial_state/velocity/x",
      state_server.imu_state.velocity(0), 0.0);
  nh.param<double>("initial_state/velocity/y",
      state_server.imu_state.velocity(1), 0.0);
  nh.param<double>("initial_state/velocity/z",
      state_server.imu_state.velocity(2), 0.0);

  // The initial covariance of orientation and position can be
  // set to 0. But for velocity, bias and extrinsic parameters,
  // there should be nontrivial uncertainty.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);

  double opti_extrinsic_rotation_cov;
  nh.param<double>("initial_covariance/opti_extrinsic_rotation_cov",
      opti_extrinsic_rotation_cov, 3.0462e-4);

  state_server.state_cov = MatrixXd::Zero(24, 24);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;
  for (int i = 21; i < 24; ++i)
    state_server.state_cov(i, i) = opti_extrinsic_rotation_cov;
  

  // Transformation offsets between the frames involved.
  Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
  Isometry3d T_cam0_imu = T_imu_cam0.inverse();

  state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
  state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();
  CAMState::T_cam0_cam1 =
    utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
  IMUState::T_imu_body =
    utils::getTransformEigen(nh, "T_imu_body").inverse();

  // Maximum number of camera states to be stored
  nh.param<int>("max_cam_state_size", max_cam_state_size, 30);

  ROS_INFO("===========================================");
  ROS_INFO("fixed frame id: %s", fixed_frame_id.c_str());
  ROS_INFO("child frame id: %s", child_frame_id.c_str());
  ROS_INFO("publish tf: %d", publish_tf);
  ROS_INFO("frame rate: %f", frame_rate);
  ROS_INFO("position std threshold: %f", position_std_threshold);
  ROS_INFO("Keyframe rotation threshold: %f", rotation_threshold);
  ROS_INFO("Keyframe translation threshold: %f", translation_threshold);
  ROS_INFO("Keyframe tracking rate threshold: %f", tracking_rate_threshold);
  ROS_INFO("gyro noise: %.10f", IMUState::gyro_noise);
  ROS_INFO("gyro bias noise: %.10f", IMUState::gyro_bias_noise);
  ROS_INFO("acc noise: %.10f", IMUState::acc_noise);
  ROS_INFO("acc bias noise: %.10f", IMUState::acc_bias_noise);
  ROS_INFO("observation noise: %.10f", Feature::observation_noise);
  ROS_INFO("initial velocity: %f, %f, %f",
      state_server.imu_state.velocity(0),
      state_server.imu_state.velocity(1),
      state_server.imu_state.velocity(2));
  ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
  ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
  ROS_INFO("initial velocity cov: %f", velocity_cov);
  ROS_INFO("initial extrinsic rotation cov: %f",
      extrinsic_rotation_cov);
  ROS_INFO("initial extrinsic translation cov: %f",
      extrinsic_translation_cov);

  std::cout << T_imu_cam0.linear() << std::endl;
  std::cout << T_imu_cam0.translation().transpose() << std::endl;

  ROS_INFO("max camera state #: %d", max_cam_state_size);
  ROS_INFO("===========================================");
  return true;
}

bool MsckfVio::createRosIO() {
  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);
  feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
      "feature_point_cloud", 10);

  reset_srv = nh.advertiseService("reset",
      &MsckfVio::resetCallback, this);

  imu_sub = nh.subscribe("imu", 100,
      &MsckfVio::imuCallback, this);

  optisensor_sub = nh.subscribe("optisensor", 100,
      &MsckfVio::optisensorCallback, this);

  optitrack_sub = nh.subscribe("optitrack", 100,
      &MsckfVio::optitrackCallback, this);

  feature_sub = nh.subscribe("features", 40,
      &MsckfVio::featureCallback, this);

  mocap_odom_sub = nh.subscribe("mocap_odom", 10,
      &MsckfVio::mocapOdomCallback, this);
  mocap_odom_pub = nh.advertise<nav_msgs::Odometry>("gt_odom", 1);

  esti_info_pub = nh.advertise<BiasEstiInfo>("bias_esti_info", 1);

  return true;
}

bool MsckfVio::initialize() {
  if (!loadParameters()) return false;
  ROS_INFO("Finish loading ROS parameters...");

  // Initialize state server
  state_server.continuous_noise_cov =
    Matrix<double, 12, 12>::Zero();
  state_server.continuous_noise_cov.block<3, 3>(0, 0) =
    Matrix3d::Identity()*IMUState::gyro_noise;
  state_server.continuous_noise_cov.block<3, 3>(3, 3) =
    Matrix3d::Identity()*IMUState::gyro_bias_noise;
  state_server.continuous_noise_cov.block<3, 3>(6, 6) =
    Matrix3d::Identity()*IMUState::acc_noise;
  state_server.continuous_noise_cov.block<3, 3>(9, 9) =
    Matrix3d::Identity()*IMUState::acc_bias_noise;

  // Initialize the chi squared test table with confidence
  // level 0.95.
  for (int i = 1; i < 100; ++i) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_test_table[i] =
      boost::math::quantile(chi_squared_dist, 0.05);
  }

  if (!createRosIO()) return false;
  ROS_INFO("Finish creating ROS IO...");

  return true;
}

void MsckfVio::optitrackCallback(const geometry_msgs::TwistStampedConstPtr &msg){
  double time = msg->header.stamp.toSec();
  Eigen::Vector3d speed;
  tf::vectorMsgToEigen(msg->twist.linear, speed);
  if(speed[0] != 0){
    speed_msg_buffer.push_back(make_pair(time, speed));
    is_first_opti = false;
  }
}

void MsckfVio::optisensorCallback(const mavros_msgs::VFR_HUDConstPtr &msg){
  double time = msg->header.stamp.toSec();
  double curr_pz = msg->climb/1000;
  if(!is_first_opti){
    double speed_y = msg->airspeed/1000/0.02;
    double speed_x = msg->groundspeed/1000/0.02;
    double speed_z = (curr_pz - prev_pz)/(time - prev_time);
    
    Eigen::Vector3d speed = Eigen::Vector3d(speed_x, speed_y, speed_z);
    if(speed_z == 0)
      speed = prev_speed;
    prev_speed = speed;
    speed_msg_buffer.push_back(make_pair(time, speed));
    cout << "speed: " << speed.transpose() << endl;
  }
  prev_pz = curr_pz;
  prev_time = time;
  is_first_opti = false;
  
}

void MsckfVio::imuCallback(
    const sensor_msgs::ImuConstPtr& msg) {

  if(!use_gt_initial){
    imu_msg_buffer.push_back(*msg);

    // if (!is_gravity_set) {
    //   if (imu_msg_buffer.size() < 200) return;
    //   // if (imu_msg_buffer.size() < 10) return;
    //   initializeGravityAndBias(msg);
    //   is_gravity_set = true;
    //   is_gt_time = true;
    // }

    if (!is_gravity_set) {
      staticinitialize(msg);
      
    }

  }else{
    // use gt 
    imu_msg_buffer.push_back(*msg);
    if (!is_gravity_set) {
      initializeGravityAndBias(msg);
    }
    
    if(msg->header.stamp.toSec() >= gt_poses[gt_num].time && !is_gt_time){
      is_gt_time = true;
      ROS_INFO("gt_num: %d, big gt_time: %f", gt_num, gt_poses[gt_num].time);
      ROS_INFO("msg_time: %f", msg->header.stamp.toSec());
    }
  }
  return;
}

bool MsckfVio::staticinitialize(const sensor_msgs::ImuConstPtr& msg){
  if (imu_msg_buffer.size() < 2) {
    return false;
  }
  bool wait_for_jerk = true;
  // Newest and oldest imu timestamp
  double newesttime = imu_msg_buffer.back().header.stamp.toSec();
  double oldesttime = imu_msg_buffer.front().header.stamp.toSec();

  // Return if we don't have enough for two windows
  if (newesttime - oldesttime < init_window_time) {
    ROS_INFO("[init-s]: unable to select window of IMU readings, not enough readings\n");
    return false;
  }

  std::vector<ImuData> window_1to0, window_2to1;
  for (const auto& imu_msg : imu_msg_buffer) {
    double time = imu_msg.header.stamp.toSec();
    ImuData data;
    data.timestamp = time;
    tf::vectorMsgToEigen(imu_msg.angular_velocity, data.wm);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, data.am);

    if (time > newesttime - 0.5 * init_window_time && time <= newesttime - 0.0 * init_window_time) {
      window_1to0.push_back(data);
    }
    if (time > newesttime - 1.0 * init_window_time && time <= newesttime - 0.5 * init_window_time) {
      window_2to1.push_back(data);
    }
  }

  // Return if both of these failed
  if (window_1to0.size() < 2 || window_2to1.size() < 2) {
    ROS_INFO("[init-s]: unable to select window of IMU readings, not enough readings\n");
    return false;
  }

  Eigen::Vector3d a_avg_1to0 = Eigen::Vector3d::Zero();
  for (const ImuData &data : window_1to0) {
    a_avg_1to0 += data.am;
  }
  a_avg_1to0 /= (int)window_1to0.size();
  double a_var_1to0 = 0;
  for (const ImuData &data : window_1to0) {
    a_var_1to0 += (data.am - a_avg_1to0).dot(data.am - a_avg_1to0);
  }
  a_var_1to0 = std::sqrt(a_var_1to0 / ((int)window_1to0.size() - 1));

  Eigen::Vector3d a_avg_2to1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d w_avg_2to1 = Eigen::Vector3d::Zero();
  for (const ImuData &data : window_2to1) {
    a_avg_2to1 += data.am;
    w_avg_2to1 += data.wm;
  }
  a_avg_2to1 = a_avg_2to1 / window_2to1.size();
  w_avg_2to1 = w_avg_2to1 / window_2to1.size();
  double a_var_2to1 = 0;

  for (const ImuData &data : window_2to1) {
    a_var_2to1 += (data.am - a_avg_2to1).dot(data.am - a_avg_2to1);
  }
  a_var_2to1 = std::sqrt(a_var_2to1 / ((int)window_2to1.size() - 1));

  if (a_var_1to0 < init_imu_thresh && wait_for_jerk) {
    ROS_INFO("[init-s]: no IMU excitation, below threshold %.3f < %.3f\n", a_var_1to0, init_imu_thresh);
    return false;
  }

  if (a_var_2to1 > init_imu_thresh && wait_for_jerk) {
    ROS_INFO("[init-s]: to much IMU excitation, above threshold %.3f > %.3f\n", a_var_2to1, init_imu_thresh);
    return false;
  }

  if ((a_var_1to0 > init_imu_thresh || a_var_2to1 > init_imu_thresh) && !wait_for_jerk) {
    ROS_INFO( "[init-s]: to much IMU excitation, above threshold %.3f,%.3f > %.3f\n", a_var_2to1, a_var_1to0,
               init_imu_thresh);
    return false;
  }

  Eigen::Vector3d z_axis = a_avg_2to1 / a_avg_2to1.norm();
  Eigen::Matrix3d Ro;
  gram_schmidt(z_axis, Ro);
  Eigen::Vector4d q_GtoI = rot_2_quat(Ro);

  // Set our biases equal to our noise (subtract our gravity from accelerometer bias)
  Eigen::Vector3d gravity_inG;
  gravity_inG << 0.0, 0.0, -IMUState::gravity(2);
  Eigen::Vector3d bg = w_avg_2to1;
  Eigen::Vector3d ba = a_avg_2to1 - quat_2_Rot(q_GtoI) * gravity_inG;
    
  Eigen::Quaterniond q0_i_w(q_GtoI);
  q0_i_w.normalize();

  state_server.imu_state.time = msg->header.stamp.toSec();
  state_server.imu_state.position = Eigen::Vector3d::Zero(); // p_imu_w
  state_server.imu_state.velocity = Eigen::Vector3d::Zero();
  state_server.imu_state.orientation = rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());
  state_server.imu_state.gyro_bias = bg;
  state_server.imu_state.acc_bias = ba;
  cout << "------------- q: " << q_GtoI.transpose() << "  bg: " << bg.transpose()
    << "  ba: " << ba.transpose() << endl;

  is_gravity_set = true;
  is_gt_time = true;
  return true;
}

void MsckfVio::initializeGravityAndBias(const sensor_msgs::ImuConstPtr& msg) {

  if(!use_gt_initial){
    // Initialize gravity and gyro bias.
    Vector3d sum_angular_vel = Vector3d::Zero();
    Vector3d sum_linear_acc = Vector3d::Zero();

    for (const auto& imu_msg : imu_msg_buffer) {
      Vector3d angular_vel = Vector3d::Zero();
      Vector3d linear_acc = Vector3d::Zero();

      tf::vectorMsgToEigen(imu_msg.angular_velocity, angular_vel);
      tf::vectorMsgToEigen(imu_msg.linear_acceleration, linear_acc);

      sum_angular_vel += angular_vel;
      sum_linear_acc += linear_acc;
    }

    state_server.imu_state.gyro_bias =
      sum_angular_vel / imu_msg_buffer.size();
    //IMUState::gravity =
    //  -sum_linear_acc / imu_msg_buffer.size();
    // This is the gravity in the IMU frame.
    Vector3d gravity_imu =
      sum_linear_acc / imu_msg_buffer.size();

    // Initialize the initial orientation, so that the estimation
    // is consistent with the inertial frame.
    double gravity_norm = gravity_imu.norm();
    IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);

    cout << "gravity_imu: " << gravity_imu.transpose() << endl;

    Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
      gravity_imu, -IMUState::gravity);

    q0_i_w.normalize();
    ROS_ERROR("gravity_norm: %f, Q0_I_W: %f, %f ,%f ,%f", gravity_norm, q0_i_w.x(), q0_i_w.y(), q0_i_w.z(), q0_i_w.w());
    state_server.imu_state.orientation =
      rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());
    return;

  }else{
    // use gt 
    double time = msg->header.stamp.toSec();
    if(gt_poses[gt_num].time > msg->header.stamp.toSec()){
      return;
    }
      
    while(gt_poses[gt_num].time < msg->header.stamp.toSec()){
      // cout << "gt_num: " << gt_num << " time: " << msg->header.stamp.toSec() 
      // << " gt_time: " << gt_poses[gt_num].time <<endl;
      gt_num ++;
    }

    if(gt_num != 0)
      gt_num --;

    ROS_INFO("gt_num: %d, gt_time: %f, position: %f, %f, %f, quat: %f, %f, %f, %f",
      gt_num, gt_poses[gt_num].time, gt_poses[gt_num].p[0], gt_poses[gt_num].p[1], gt_poses[gt_num].p[2],
      gt_poses[gt_num].q.x(), gt_poses[gt_num].q.y(), gt_poses[gt_num].q.z(), gt_poses[gt_num].q.w());

    gt_init = gt_num; 
    
    state_server.imu_state.time = gt_poses[gt_init].time;
    std::cout << "gt_init: " << gt_init << std::endl;
    std::cout << gt_poses[gt_init].p.transpose() << std::endl;
    state_server.imu_state.orientation =
      rotationToQuaternion(gt_poses[gt_init].q.toRotationMatrix().cast<double>().transpose()); //T_w_imu
    state_server.imu_state.position = gt_poses[gt_init].p.cast<double>(); //p_imu_w
    state_server.imu_state.velocity = gt_poses[gt_init].vel.cast<double>();
    is_gravity_set = true;
    return;
  }
}

bool MsckfVio::resetCallback(
    std_srvs::Trigger::Request& req,
    std_srvs::Trigger::Response& res) {

  ROS_WARN("Start resetting msckf vio...");
  // Temporarily shutdown the subscribers to prevent the
  // state from updating.
  feature_sub.shutdown();
  imu_sub.shutdown();

  // Reset the IMU state.
  IMUState& imu_state = state_server.imu_state;
  imu_state.time = 0.0;
  imu_state.orientation = Vector4d(0.0, 0.0, 0.0, 1.0);
  imu_state.position = Vector3d::Zero();
  imu_state.velocity = Vector3d::Zero();
  imu_state.gyro_bias = Vector3d::Zero();
  imu_state.acc_bias = Vector3d::Zero();
  imu_state.orientation_null = Vector4d(0.0, 0.0, 0.0, 1.0);
  imu_state.position_null = Vector3d::Zero();
  imu_state.velocity_null = Vector3d::Zero();

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);
  
  double opti_extrinsic_rotation_cov;
  nh.param<double>("initial_covariance/opti_extrinsic_rotation_cov",
      opti_extrinsic_rotation_cov, 3.0462e-4);

  state_server.state_cov = MatrixXd::Zero(24, 24);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;
  for (int i = 21; i < 24; ++i)
    state_server.state_cov(i, i) = opti_extrinsic_rotation_cov;

  // Clear all exsiting features in the map.
  map_server.clear();

  // Clear the IMU msg buffer.
  imu_msg_buffer.clear();

  // Reset the starting flags.
  is_gravity_set = false;
  is_first_img = true;

  // Restart the subscribers.
  imu_sub = nh.subscribe("imu", 100,
      &MsckfVio::imuCallback, this);
  feature_sub = nh.subscribe("features", 40,
      &MsckfVio::featureCallback, this);

  // TODO: When can the reset fail?
  res.success = true;
  ROS_WARN("Resetting msckf vio completed...");
  return true;
}

void MsckfVio::featureCallback(
    const CameraMeasurementConstPtr& msg) {

  // Return if the gravity vector has not been set.
  if (!is_gt_time) return;

  // Start the system if the first image is received.
  // The frame where the first image is received will be
  // the origin.
  if (is_first_img) {
    is_first_img = false;
    state_server.imu_state.time = msg->header.stamp.toSec();
  }

  static double max_processing_time = 0.0;
  static int critical_time_cntr = 0;
  double processing_start_time = ros::Time::now().toSec();

  if(!finish_initialize_optiflow && window.size() == 40){
    initialize_optiflow();
    finish_initialize_optiflow = true;
  }

  estiImuBias(msg->header.stamp.toSec());
  // ROS_INFO("finish estiImuBias! ");

  // Propogate the IMU state.
  // that are received before the image msg.
  ros::Time start_time = ros::Time::now();
  batchImuProcessing(msg->header.stamp.toSec());
  double imu_processing_time = (
      ros::Time::now()-start_time).toSec();
  // ROS_INFO("finish batchImuProcessing! ");

  // Augment the state vector.
  start_time = ros::Time::now();
  stateAugmentation(msg->header.stamp.toSec());
  double state_augmentation_time = (
      ros::Time::now()-start_time).toSec();
  // ROS_INFO("finish stateAugmentation! ");

  // Add new observations for existing features or new
  // features in the map server.
  start_time = ros::Time::now();
  addFeatureObservations(msg);
  double add_observations_time = (
      ros::Time::now()-start_time).toSec();
  // ROS_INFO("finish addFeatureObservations! ");
  
  
  // Perform measurement update if necessary.
  start_time = ros::Time::now();
  removeLostFeatures();
  double remove_lost_features_time = (
      ros::Time::now()-start_time).toSec();
  // ROS_INFO("finish removeLostFeatures! ");

  start_time = ros::Time::now();
  pruneCamStateBuffer();
  double prune_cam_states_time = (
      ros::Time::now()-start_time).toSec();
  // ROS_INFO("finish pruneCamStateBuffer! ");
  
  if(finish_initialize_optiflow){
    optiflowProcess();
    // ROS_INFO("optiflowProcess! ");
  }

  // Publish the odometry.
  start_time = ros::Time::now();
  publish(msg->header.stamp);
  double publish_time = (
      ros::Time::now()-start_time).toSec(); 

  // Reset the system if necessary.
  onlineReset();

  double processing_end_time = ros::Time::now().toSec();
  double processing_time =
    processing_end_time - processing_start_time;
  if (processing_time > 1.0/frame_rate) {
    ++critical_time_cntr;
    ROS_INFO("\033[1;31mTotal processing time %f/%d...\033[0m",
        processing_time, critical_time_cntr);
    //printf("IMU processing time: %f/%f\n",
    //    imu_processing_time, imu_processing_time/processing_time);
    //printf("State augmentation time: %f/%f\n",
    //    state_augmentation_time, state_augmentation_time/processing_time);
    //printf("Add observations time: %f/%f\n",
    //    add_observations_time, add_observations_time/processing_time);
    // printf("Remove lost features time: %f/%f\n",
    //     remove_lost_features_time, remove_lost_features_time/processing_time);
    // printf("Remove camera states time: %f/%f\n",
    //     prune_cam_states_time, prune_cam_states_time/processing_time);
    //printf("Publish time: %f/%f\n",
    //    publish_time, publish_time/processing_time);
  }

  return;
}

void MsckfVio::initialize_optiflow(){
  ROS_ERROR(" START INITIALIZE !");
  ceres::Problem problem;
  double q_imu_opti[4] = {0, 0, 0, 1};

  ceres::LossFunction *loss_function;
  loss_function = new ceres::HuberLoss(1.0);
  ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<InitializeOpti, 120, 4>(
        new InitializeOpti(window));     
  problem.AddResidualBlock(cost_function, loss_function, q_imu_opti);

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;

  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.trust_region_strategy_type = ceres::DOGLEG;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  Eigen::Quaterniond q(Eigen::Vector4d(q_imu_opti[0], q_imu_opti[1], q_imu_opti[2], q_imu_opti[3]));
  q.normalize();
  ROS_ERROR("q: %f, %f, %f ,%f", q.x(), q.y(), q.z(), q.w());
  
  cout << "q_imu_opti: " << q.toRotationMatrix() << endl;

  state_server.imu_state.R_imu_opti = q.toRotationMatrix();

  state_server.imu_state.R_imu_opti << 0.93937,  -0.340883,   0.037192,
  0.333882,   0.933972,   0.127358,
-0.0781504,  -0.107218,   0.991159;
}

void MsckfVio::OptiflowmeasurementUpdate(const Eigen::MatrixXd& H,const Eigen::VectorXd&r,const Eigen::MatrixXd &noise)
{
  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  
  if (H.rows() > H.cols()) {
      // Convert H to a sparse matrix.
      SparseMatrix<double> H_sparse = H.sparseView();

      // Perform QR decomposition on H_sparse.
      SPQR<SparseMatrix<double> > spqr_helper;
      spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
      spqr_helper.compute(H_sparse);

      MatrixXd H_temp;
      VectorXd r_temp;
      (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
      (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

      H_thin = H_temp.topRows(24+state_server.cam_states.size()*6);
      r_thin = r_temp.head(24+state_server.cam_states.size()*6);

      //HouseholderQR<MatrixXd> qr_helper(H);
      //MatrixXd Q = qr_helper.householderQ();
      //MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

      //H_thin = Q1.transpose() * H;
      //r_thin = Q1.transpose() * r;
  } else {
      H_thin = H;
      r_thin = r;
  }
  // cout << H << endl;
  // Compute the Kalman gain.
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() + noise;
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // Update the IMU state.
  const VectorXd& delta_x_imu = delta_x.head<24>();

  if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
          //delta_x_imu.segment<3>(3).norm() > 0.15 ||
          delta_x_imu.segment<3>(6).norm() > 0.5 ||
          //delta_x_imu.segment<3>(9).norm() > 0.5 ||
          delta_x_imu.segment<3>(12).norm() > 1.0) {
      printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
      printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
      ROS_WARN("Update change is too large.");
      //return;
  }

  const Vector4d dq_imu =
          smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
              dq_imu, state_server.imu_state.orientation);
  // cout << "delta dq_imu: " << delta_x_imu.head<3>() << endl;

  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
  // cout << "delta gyro_bias: " << delta_x_imu.segment<3>(3) << endl;

  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  // cout << "delta velocity: " << delta_x_imu.segment<3>(6) << endl;

  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
  // cout << "delta acc_bias: " << delta_x_imu.segment<3>(9) << endl;

  state_server.imu_state.position += delta_x_imu.segment<3>(12);
  // cout << "delta position: " << delta_x_imu.segment<3>(12) << endl;

  const Vector4d dq_extrinsic =
          smallAngleQuaternion(delta_x_imu.segment<3>(15));
  // cout << "dq_extrinsic: " << delta_x_imu.segment<3>(15) << endl;

  state_server.imu_state.R_imu_cam0 = quaternionToRotation(
              dq_extrinsic) * state_server.imu_state.R_imu_cam0;  
  state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);
  // cout << "dt_extrinsic: " << delta_x_imu.segment<3>(18) << endl;


  const Vector4d dq_extrinsic_opti =
          smallAngleQuaternion(delta_x_imu.segment<3>(21));
  // cout << "dq_extrinsic_opti: " << delta_x_imu.segment<3>(15) << endl;

  state_server.imu_state.R_imu_opti = quaternionToRotation(
              dq_extrinsic_opti) * state_server.imu_state.R_imu_opti;

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
        ++i, ++cam_state_iter) {
      const VectorXd& delta_x_cam = delta_x.segment<6>(24+i*6);
      const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
      cam_state_iter->second.orientation = quaternionMultiplication(
                  dq_cam, cam_state_iter->second.orientation);
      cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
  //  K*K.transpose()*Feature::observation_noise;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
                              state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

void MsckfVio::optiflowProcess(){  

  Eigen::MatrixXd H_x = MatrixXd::Zero(3,24+6*state_server.cam_states.size());
  Eigen::VectorXd r = VectorXd::Zero(3);

  IMUState &imu_state = state_server.imu_state;

  H_x.block<3,3>(0,6) = imu_state.R_imu_opti;// q bg v ba p
  H_x.block<3,3>(0,21) = -skewSymmetric(imu_state.R_imu_opti * imu_state.velocity);

  // H_x.block<3,3>(0,6) = Eigen::Matrix3d::Identity();// q bg v ba p

  r.segment<3>(0) = imu_state.opti_speed - imu_state.R_imu_opti * imu_state.velocity;
  cout << "r: " << r.transpose() << " opti: " << imu_state.opti_speed.transpose() 
    << " vel: " << imu_state.velocity.transpose() << endl;

  Eigen::MatrixXd noise = Eigen::MatrixXd::Identity(3, 3) * 0.1;

  OptiflowmeasurementUpdate(H_x, r, noise);

  cout << "R_imu_opti: " << imu_state.R_imu_opti << endl;

}

void MsckfVio::estiImuBias(const double time){

  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  int num = gt_num;
  while(num < gt_poses.size()){
    if (gt_poses[num].time <= state_server.imu_state.time){
        num ++;
    }else{
        break;
    }
  }

  if(num > 1){
    int prev_num = num - 2;
    int next_num = num + 2;
    double prev_time = gt_poses[prev_num].time;
    double curr_time = gt_poses[next_num].time;
    Eigen::Vector3d prev_p = gt_poses[prev_num].p;
    Eigen::Vector3d curr_p = gt_poses[next_num].p;

    // while(curr_time - prev_time < 1e-2 && prev_num > 0){
    //   prev_num--;
    //   prev_time = gt_poses[prev_num].time;
    //   prev_p = gt_poses[prev_num].p;
    // }

    vel = (curr_p - prev_p) / (curr_time - prev_time);
  }
  state_server.imu_state.opti_speed = vel;

  cout << "speed_msg_buffer.size(): " << speed_msg_buffer.size() << endl;
  if(speed_msg_buffer.size() <= 0)
    return;

  IMUState& imu_state = state_server.imu_state;
  int used_msg = 0;
  double time_bound = imu_state.time;
  int id = 0;
  for(; id < speed_msg_buffer.size(); id++){
    double msg_time = speed_msg_buffer[id].first;
    if (msg_time < time_bound) {
      ++used_msg;
      continue;
    }

    if (msg_time > time) {
      break;
    }  
    ++used_msg;
  }

  // linear 
  Eigen::Vector3d opti_speed;
  if(id > 0){
    Eigen::Vector3d prev_speed = speed_msg_buffer[id-1].second;
    double prev_time = speed_msg_buffer[id-1].first;

    Eigen::Vector3d curr_speed = speed_msg_buffer[id].second;
    double curr_time = speed_msg_buffer[id].first;
    opti_speed = prev_speed + (curr_speed - prev_speed) / (curr_time - prev_time) * (time - prev_time);
  }else
    opti_speed = speed_msg_buffer[id].second;

  imu_state.opti_speed = opti_speed;
  cout << "opti_speed: " << opti_speed.transpose() << endl;
  // if(window.size() > 0){
  //   Eigen::Vector3d sum_opti_speed = opti_speed;

  //   auto iter = window.rbegin();
  //   int count = 0;
  //   while(iter != window.rend() && count < 3){
  //     sum_opti_speed += iter->opti_speed;
  //     count ++;
  //     iter ++;
  //   }
  //   cout << "count :" << count << endl;
  //   imu_state.opti_speed = sum_opti_speed / (count+1);
  // }else 
  //   imu_state.opti_speed = opti_speed;

  speed_msg_buffer.erase(speed_msg_buffer.begin(),
      speed_msg_buffer.begin()+used_msg);

  return;
}


void MsckfVio::mocapOdomCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  static bool first_mocap_odom_msg = true;

  // If this is the first mocap odometry messsage, set
  // the initial frame.
  if (first_mocap_odom_msg) {
    Quaterniond orientation;
    Vector3d translation;
    tf::pointMsgToEigen(
        msg->pose.pose.position, translation);
    tf::quaternionMsgToEigen(
        msg->pose.pose.orientation, orientation);
    //tf::vectorMsgToEigen(
    //    msg->transform.translation, translation);
    //tf::quaternionMsgToEigen(
    //    msg->transform.rotation, orientation);
    mocap_initial_frame.linear() = orientation.toRotationMatrix();
    mocap_initial_frame.translation() = translation;
    first_mocap_odom_msg = false;
  }

  // Transform the ground truth.
  Quaterniond orientation;
  Vector3d translation;
  //tf::vectorMsgToEigen(
  //    msg->transform.translation, translation);
  //tf::quaternionMsgToEigen(
  //    msg->transform.rotation, orientation);
  tf::pointMsgToEigen(
      msg->pose.pose.position, translation);
  tf::quaternionMsgToEigen(
      msg->pose.pose.orientation, orientation);

  Eigen::Isometry3d T_b_v_gt;
  T_b_v_gt.linear() = orientation.toRotationMatrix();
  T_b_v_gt.translation() = translation;
  Eigen::Isometry3d T_b_w_gt = mocap_initial_frame.inverse() * T_b_v_gt;

  //Eigen::Vector3d body_velocity_gt;
  //tf::vectorMsgToEigen(msg->twist.twist.linear, body_velocity_gt);
  //body_velocity_gt = mocap_initial_frame.linear().transpose() *
  //  body_velocity_gt;

  // Ground truth tf.
  if (publish_tf) {
    tf::Transform T_b_w_gt_tf;
    tf::transformEigenToTF(T_b_w_gt, T_b_w_gt_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_b_w_gt_tf, msg->header.stamp, fixed_frame_id, child_frame_id+"_mocap"));
  }

  // Ground truth odometry.
  nav_msgs::Odometry mocap_odom_msg;
  mocap_odom_msg.header.stamp = msg->header.stamp;
  mocap_odom_msg.header.frame_id = fixed_frame_id;
  mocap_odom_msg.child_frame_id = child_frame_id+"_mocap";

  tf::poseEigenToMsg(T_b_w_gt, mocap_odom_msg.pose.pose);
  //tf::vectorEigenToMsg(body_velocity_gt,
  //    mocap_odom_msg.twist.twist.linear);

  mocap_odom_pub.publish(mocap_odom_msg);
  return;
}

void MsckfVio::batchImuProcessing(const double& time_bound) {
  // Counter how many IMU msgs in the buffer are used.
  int used_imu_msg_cntr = 0;
  use_imu_num = 0;
  // cout << "gravity: " << IMUState::gravity << endl;
  for (const auto& imu_msg : imu_msg_buffer) {
    double imu_time = imu_msg.header.stamp.toSec();
    if (imu_time < state_server.imu_state.time) {
      ++used_imu_msg_cntr;
      continue;
    }
    if (imu_time > time_bound) break;

    // Convert the msgs.
    Vector3d m_gyro, m_acc;
    tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);
    
    // Execute process model.
    processModel(imu_time, m_gyro, m_acc);
    ++used_imu_msg_cntr;
    use_imu_num ++;
  }

  // Set the state ID for the new IMU state.
  state_server.imu_state.id = IMUState::next_id++;

  // Remove all used IMU msgs.
  imu_msg_buffer.erase(imu_msg_buffer.begin(),
      imu_msg_buffer.begin()+used_imu_msg_cntr);

  return;
}

void MsckfVio::processModel(const double& time,
    const Vector3d& m_gyro,
    const Vector3d& m_acc) {

  // Remove the bias from the measured gyro and acceleration
  IMUState& imu_state = state_server.imu_state;
  Vector3d gyro = m_gyro - imu_state.gyro_bias;
  Vector3d acc = m_acc - imu_state.acc_bias;
  double dtime = time - imu_state.time;

  // Compute discrete transition and noise covariance matrix
  Matrix<double, 24, 24> F = Matrix<double, 24, 24>::Zero();
  Matrix<double, 24, 12> G = Matrix<double, 24, 12>::Zero();

  F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  F.block<3, 3>(0, 3) = -Matrix3d::Identity();
  F.block<3, 3>(6, 0) = -quaternionToRotation(
      imu_state.orientation).transpose()*skewSymmetric(acc);
  F.block<3, 3>(6, 9) = -quaternionToRotation(
      imu_state.orientation).transpose();
  F.block<3, 3>(12, 6) = Matrix3d::Identity();

  G.block<3, 3>(0, 0) = -Matrix3d::Identity();
  G.block<3, 3>(3, 3) = Matrix3d::Identity();
  G.block<3, 3>(6, 6) = -quaternionToRotation(
      imu_state.orientation).transpose();
  G.block<3, 3>(9, 9) = Matrix3d::Identity();

  // Approximate matrix exponential to the 3rd order,
  // which can be considered to be accurate enough assuming
  // dtime is within 0.01s.
  Matrix<double, 24, 24> Fdt = F * dtime;
  Matrix<double, 24, 24> Fdt_square = Fdt * Fdt;
  Matrix<double, 24, 24> Fdt_cube = Fdt_square * Fdt;
  Matrix<double, 24, 24> Phi = Matrix<double, 24, 24>::Identity() +
    Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

  // Propogate the state using 4th order Runge-Kutta
  predictNewState(dtime, gyro, acc);

  // Modify the transition matrix
  Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
  Phi.block<3, 3>(0, 0) =
    quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

  Vector3d u = R_kk_1 * IMUState::gravity;
  RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

  Matrix3d A1 = Phi.block<3, 3>(6, 0);
  Vector3d w1 = skewSymmetric(
      imu_state.velocity_null-imu_state.velocity) * IMUState::gravity;
  Phi.block<3, 3>(6, 0) = A1 - (A1*u-w1)*s;

  Matrix3d A2 = Phi.block<3, 3>(12, 0);
  Vector3d w2 = skewSymmetric(
      dtime*imu_state.velocity_null+imu_state.position_null-
      imu_state.position) * IMUState::gravity;
  Phi.block<3, 3>(12, 0) = A2 - (A2*u-w2)*s;

  // Propogate the state covariance matrix.
  Matrix<double, 24, 24> Q = Phi*G*state_server.continuous_noise_cov*
    G.transpose()*Phi.transpose()*dtime;
  state_server.state_cov.block<24, 24>(0, 0) =
    Phi*state_server.state_cov.block<24, 24>(0, 0)*Phi.transpose() + Q;

  if (state_server.cam_states.size() > 0) {
    state_server.state_cov.block(
        0, 24, 24, state_server.state_cov.cols()-24) =
      Phi * state_server.state_cov.block(
        0, 24, 24, state_server.state_cov.cols()-24);
    state_server.state_cov.block(
        24, 0, state_server.state_cov.rows()-24, 24) =
      state_server.state_cov.block(
        24, 0, state_server.state_cov.rows()-24, 24) * Phi.transpose();
  }

  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  // Update the state correspondes to null space.
  imu_state.orientation_null = imu_state.orientation;
  imu_state.position_null = imu_state.position;
  imu_state.velocity_null = imu_state.velocity;

  // Update the state info
  state_server.imu_state.time = time;
  return;
}

void MsckfVio::predictNewState(const double& dt,
    const Vector3d& gyro,
    const Vector3d& acc) {

  // TODO: Will performing the forward integration using
  //    the inverse of the quaternion give better accuracy?
  double gyro_norm = gyro.norm();
  Matrix4d Omega = Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro;

  Vector4d& q = state_server.imu_state.orientation;
  Vector3d& v = state_server.imu_state.velocity;
  Vector3d& p = state_server.imu_state.position;

  // Some pre-calculation
  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
  }
  else {
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
      cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
      cos(gyro_norm*dt*0.25) * q;
  }
  Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
  Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

  // k1 = f(tn, yn)
  Vector3d k1_v_dot = quaternionToRotation(q).transpose()*acc +
    IMUState::gravity;
  // cout << "k1_v_dot: " << k1_v_dot.transpose() << endl;

  Vector3d k1_p_dot = v;

  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k2_p_dot = k1_v;

  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k3_p_dot = k2_v;

  // k4 = f(tn+dt, yn+k3*dt)
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose*acc +
    IMUState::gravity;
  Vector3d k4_p_dot = k3_v;

  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = dq_dt;
  quaternionNormalize(q);
  v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

  return;
}

void MsckfVio::stateAugmentation(const double& time) {

  const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;

  // Add a new camera state to the state server.
  Matrix3d R_w_i = quaternionToRotation(
      state_server.imu_state.orientation);
  Matrix3d R_w_c = R_i_c * R_w_i;
  Vector3d t_c_w = state_server.imu_state.position +
    R_w_i.transpose()*t_c_i;

  state_server.cam_states[state_server.imu_state.id] =
    CAMState(state_server.imu_state.id);
  CAMState& cam_state = state_server.cam_states[
    state_server.imu_state.id];

  cam_state.time = time;
  cam_state.orientation = rotationToQuaternion(R_w_c);
  cam_state.position = t_c_w;  


  cam_state.orientation_null = cam_state.orientation;
  cam_state.position_null = cam_state.position;

  // Update the covariance matrix of the state.
  // To simplify computation, the matrix J below is the nontrivial block
  // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
  // -aided Inertial Navigation".
  Matrix<double, 6, 24> J = Matrix<double, 6, 24>::Zero();
  J.block<3, 3>(0, 0) = R_i_c;
  J.block<3, 3>(0, 15) = Matrix3d::Identity();
  J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose()*t_c_i);
  //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
  J.block<3, 3>(3, 12) = Matrix3d::Identity();
  J.block<3, 3>(3, 18) = R_w_i.transpose();

  // Resize the state covariance matrix.
  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();
  state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Rename some matrix blocks for convenience.
  const Matrix<double, 24, 24>& P11 =
    state_server.state_cov.block<24, 24>(0, 0);
  const MatrixXd& P12 =
    state_server.state_cov.block(0, 24, 24, old_cols-24);

  // Fill in the augmented state covariance.
  state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
  state_server.state_cov.block(0, old_cols, old_rows, 6) =
    state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
  state_server.state_cov.block<6, 6>(old_rows, old_cols) =
    J * P11 * J.transpose();

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

void MsckfVio::addFeatureObservations(
    const CameraMeasurementConstPtr& msg) {

  StateIDType state_id = state_server.imu_state.id;
  int curr_feature_num = map_server.size();
  int tracked_feature_num = 0;

  // Add new observations for existing features or new
  // features in the map server.
  for (const auto& feature : msg->features) {
    if (map_server.find(feature.id) == map_server.end()) {
      // This is a new feature.
      map_server[feature.id] = Feature(feature.id);
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.u0, feature.v0,
            feature.u1, feature.v1);
    } else {
      // This is an old feature.
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.u0, feature.v0,
            feature.u1, feature.v1);
      ++tracked_feature_num;
    }
  }

  tracking_rate =
    static_cast<double>(tracked_feature_num) /
    static_cast<double>(curr_feature_num);

  return;
}

void MsckfVio::measurementJacobian(
    const StateIDType& cam_state_id,
    const FeatureIDType& feature_id,
    Matrix<double, 4, 6>& H_x, Matrix<double, 4, 3>& H_f, Vector4d& r) {

  // Prepare all the required data.
  const CAMState& cam_state = state_server.cam_states[cam_state_id];
  const Feature& feature = map_server[feature_id];

  // Cam0 pose.
  Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
  const Vector3d& t_c0_w = cam_state.position;

  // Cam1 pose.
  Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
  Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
  Vector3d t_c1_w = t_c0_w - R_w_c1.transpose()*CAMState::T_cam0_cam1.translation();

  // 3d feature position in the world frame.
  // And its observation with the stereo cameras.
  const Vector3d& p_w = feature.position;
  const Vector4d& z = feature.observations.find(cam_state_id)->second;

  // Convert the feature position from the world frame to
  // the cam0 and cam1 frame.
  Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
  Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);

  // Compute the Jacobians.
  Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
  dz_dpc0(0, 0) = 1 / p_c0(2);
  dz_dpc0(1, 1) = 1 / p_c0(2);
  dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
  dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

  Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
  dz_dpc1(2, 0) = 1 / p_c1(2);
  dz_dpc1(3, 1) = 1 / p_c1(2);
  dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
  dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

  Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
  dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
  dpc0_dxc.rightCols(3) = -R_w_c0;

  Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
  dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
  dpc1_dxc.rightCols(3) = -R_w_c1;

  Matrix3d dpc0_dpg = R_w_c0;
  Matrix3d dpc1_dpg = R_w_c1;

  H_x = dz_dpc0*dpc0_dxc + dz_dpc1*dpc1_dxc;
  H_f = dz_dpc0*dpc0_dpg + dz_dpc1*dpc1_dpg;

  // Modifty the measurement Jacobian to ensure
  // observability constrain.
  Matrix<double, 4, 6> A = H_x;
  Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
  u.block<3, 1>(0, 0) = quaternionToRotation(
      cam_state.orientation_null) * IMUState::gravity;
  u.block<3, 1>(3, 0) = skewSymmetric(
      p_w-cam_state.position_null) * IMUState::gravity;
  H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
  H_f = -H_x.block<4, 3>(0, 3);

  // Compute the residual.
  r = z - Vector4d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2),
      p_c1(0)/p_c1(2), p_c1(1)/p_c1(2));

  return;
}

void MsckfVio::featureJacobian(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) {

  const auto& feature = map_server[feature_id];

  // Check how many camera states in the provided camera
  // id camera has actually seen this feature.
  vector<StateIDType> valid_cam_state_ids(0);
  for (const auto& cam_id : cam_state_ids) {
    if (feature.observations.find(cam_id) ==
        feature.observations.end()) continue;

    valid_cam_state_ids.push_back(cam_id);
  }

  int jacobian_row_size = 0;
  jacobian_row_size = 4 * valid_cam_state_ids.size();

  MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
      24+state_server.cam_states.size()*6);
  MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
  VectorXd r_j = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (const auto& cam_id : valid_cam_state_ids) {

    Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
    Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
    Vector4d r_i = Vector4d::Zero();
    measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

    auto cam_state_iter = state_server.cam_states.find(cam_id);
    int cam_state_cntr = std::distance(
        state_server.cam_states.begin(), cam_state_iter);

    // Stack the Jacobians.
    H_xj.block<4, 6>(stack_cntr, 24+6*cam_state_cntr) = H_xi;
    H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
    r_j.segment<4>(stack_cntr) = r_i;
    stack_cntr += 4;
  }

  // Project the residual and Jacobians onto the nullspace
  // of H_fj.
  JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
  MatrixXd A = svd_helper.matrixU().rightCols(
      jacobian_row_size - 3);

  H_x = A.transpose() * H_xj;
  r = A.transpose() * r_j;

  return;
}

void MsckfVio::measurementUpdate(
    const MatrixXd& H, const VectorXd& r) {

  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(24+state_server.cam_states.size()*6);
    r_thin = r_temp.head(24+state_server.cam_states.size()*6);

    //HouseholderQR<MatrixXd> qr_helper(H);
    //MatrixXd Q = qr_helper.householderQ();
    //MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

    //H_thin = Q1.transpose() * H;
    //r_thin = Q1.transpose() * r;
  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() +
      Feature::observation_noise*MatrixXd::Identity(
        H_thin.rows(), H_thin.rows());
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // Update the IMU state.
  const VectorXd& delta_x_imu = delta_x.head<24>();

  if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
      //delta_x_imu.segment<3>(3).norm() > 0.15 ||
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      //delta_x_imu.segment<3>(9).norm() > 0.5 ||
      delta_x_imu.segment<3>(12).norm() > 1.0) {
    printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
    printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
    ROS_WARN("Update change is too large.");
    //return;
  }

  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);

  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
  // cout << "delta gyro_bias: " << delta_x_imu.segment<3>(3) << endl;

  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
  // cout << "delta acc_bias: " << delta_x_imu.segment<3>(9) << endl;

  state_server.imu_state.position += delta_x_imu.segment<3>(12);
  // cout << "delta position: " << delta_x_imu.segment<3>(12) << endl;

  const Vector4d dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(15));
  state_server.imu_state.R_imu_cam0 = quaternionToRotation(
      dq_extrinsic) * state_server.imu_state.R_imu_cam0;
  state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

  const Vector4d dq_extrinsic_opti =
          smallAngleQuaternion(delta_x_imu.segment<3>(21));
  // cout << "dq_extrinsic_opti: " << delta_x_imu.segment<3>(15) << endl;

  state_server.imu_state.R_imu_opti = quaternionToRotation(
              dq_extrinsic_opti) * state_server.imu_state.R_imu_opti;

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(24+i*6);
    const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq_cam, cam_state_iter->second.orientation);
    cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
  //  K*K.transpose()*Feature::observation_noise;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

bool MsckfVio::gatingTest(
    const MatrixXd& H, const VectorXd& r, const int& dof) {

  MatrixXd P1 = H * state_server.state_cov * H.transpose();
  MatrixXd P2 = Feature::observation_noise *
    MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

  //cout << dof << " " << gamma << " " <<
  //  chi_squared_test_table[dof] << " ";

  if (gamma < chi_squared_test_table[dof]) {
    //cout << "passed" << endl;
    return true;
  } else {
    //cout << "failed" << endl;
    return false;
  }
}

void MsckfVio::removeLostFeatures() {

  // Remove the features that lost track.
  // BTW, find the size the final Jacobian matrix and residual vector.
  int jacobian_row_size = 0;
  vector<FeatureIDType> invalid_feature_ids(0);
  vector<FeatureIDType> processed_feature_ids(0);

  for (auto iter = map_server.begin();
      iter != map_server.end(); ++iter) {
    // Rename the feature to be checked.
    auto& feature = iter->second;

    // Pass the features that are still being tracked.
    if (feature.observations.find(state_server.imu_state.id) !=
        feature.observations.end()) continue;
    if (feature.observations.size() < 3) {
      invalid_feature_ids.push_back(feature.id);
      continue;
    }

    // Check if the feature can be initialized if it
    // has not been.
    if (!feature.is_initialized) {
      if (!feature.checkMotion(state_server.cam_states)) {
        invalid_feature_ids.push_back(feature.id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          invalid_feature_ids.push_back(feature.id);
          continue;
        }
      }
    }

    jacobian_row_size += 4*feature.observations.size() - 3;
    processed_feature_ids.push_back(feature.id);
  }

  //cout << "invalid/processed feature #: " <<
  //  invalid_feature_ids.size() << "/" <<
  //  processed_feature_ids.size() << endl;
  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Remove the features that do not have enough measurements.
  for (const auto& feature_id : invalid_feature_ids)
    map_server.erase(feature_id);

  // Return if there is no lost feature to be processed.
  if (processed_feature_ids.size() == 0) return;

  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      24+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  // Process the features which lose track.
  for (const auto& feature_id : processed_feature_ids) {
    auto& feature = map_server[feature_id];

    vector<StateIDType> cam_state_ids(0);
    for (const auto& measurement : feature.observations)
      cam_state_ids.push_back(measurement.first);

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    // Put an upper bound on the row size of measurement Jacobian,
    // which helps guarantee the executation time.
    if (stack_cntr > 1500) break;
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform the measurement update step.
  measurementUpdate(H_x, r);

  // Remove all processed features from the map.
  for (const auto& feature_id : processed_feature_ids)
    map_server.erase(feature_id);

  return;
}

void MsckfVio::findRedundantCamStates(
    vector<StateIDType>& rm_cam_state_ids, bool& remove_first) {

  // Move the iterator to the key position.
  auto key_cam_state_iter = state_server.cam_states.end();
  for (int i = 0; i < 4; ++i)
    --key_cam_state_iter; // 倒数第四帧
  auto cam_state_iter = key_cam_state_iter;
  ++cam_state_iter; // 倒数第三帧
  auto first_cam_state_iter = state_server.cam_states.begin();// 第一帧

  // Pose of the key camera state.
  const Vector3d key_position = // 倒数第四帧
    key_cam_state_iter->second.position;
  const Matrix3d key_rotation = quaternionToRotation(
      key_cam_state_iter->second.orientation);

  // Mark the camera states to be removed based on the
  // motion between states.
  for (int i = 0; i < 2; ++i) {
    const Vector3d position =
      cam_state_iter->second.position; // 倒数第三帧
    const Matrix3d rotation = quaternionToRotation(
        cam_state_iter->second.orientation);

    double distance = (position-key_position).norm();
    double angle = AngleAxisd(
        rotation*key_rotation.transpose()).angle();

    if (angle < rotation_threshold &&
        distance < translation_threshold &&
        tracking_rate > tracking_rate_threshold) {
      rm_cam_state_ids.push_back(cam_state_iter->first);
      ++cam_state_iter;
      // ROS_INFO("remove last frame!");
      remove_first = false;
    } else {
      rm_cam_state_ids.push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
      // ROS_INFO("remove first frame!");
      remove_first = true;
      has_remove_state = true;
    }
  }

  // Sort the elements in the output vector.
  sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

  return;
}

void MsckfVio::pruneCamStateBuffer() {

  if (state_server.cam_states.size() < max_cam_state_size)
    return;

  // Find two camera states to be removed.
  vector<StateIDType> rm_cam_state_ids(0);
  bool remove_first = true;
  findRedundantCamStates(rm_cam_state_ids, remove_first);

  // Find the size of the Jacobian matrix.
  int jacobian_row_size = 0;
  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;
    if (involved_cam_state_ids.size() == 1) {
      feature.observations.erase(involved_cam_state_ids[0]);
      continue;
    }

    if (!feature.is_initialized) {
      // Check if the feature can be initialize.
      if (!feature.checkMotion(state_server.cam_states)) {
        // If the feature cannot be initialized, just remove
        // the observations associated with the camera states
        // to be removed.
        for (const auto& cam_id : involved_cam_state_ids)
          feature.observations.erase(cam_id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
          continue;
        }
      }
    }

    jacobian_row_size += 4*involved_cam_state_ids.size() - 3;
  }

  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Compute the Jacobian and residual.
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      24+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    for (const auto& cam_id : involved_cam_state_ids)
      feature.observations.erase(cam_id);
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform measurement update.
  measurementUpdate(H_x, r);

  for (const auto& cam_id : rm_cam_state_ids) {
    int cam_sequence = std::distance(state_server.cam_states.begin(),
        state_server.cam_states.find(cam_id));
    int cam_state_start = 24 + 6*cam_sequence;
    int cam_state_end = cam_state_start + 6;

    // Remove the corresponding rows and columns in the state
    // covariance matrix.
    if (cam_state_end < state_server.state_cov.rows()) {
      state_server.state_cov.block(cam_state_start, 0,
          state_server.state_cov.rows()-cam_state_end,
          state_server.state_cov.cols()) =
        state_server.state_cov.block(cam_state_end, 0,
            state_server.state_cov.rows()-cam_state_end,
            state_server.state_cov.cols());

      state_server.state_cov.block(0, cam_state_start,
          state_server.state_cov.rows(),
          state_server.state_cov.cols()-cam_state_end) =
        state_server.state_cov.block(0, cam_state_end,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-cam_state_end);

      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    } else {
      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    }  
  }

  std::vector<std::pair<double, Eigen::Vector3d>> acc_set;
  std::vector<std::pair<double, Eigen::Vector3d>> gyro_set;
  CAMState &curr_camstate = state_server.cam_states[state_server.imu_state.id];

  for (const auto& cam_id : rm_cam_state_ids) {
    state_server.cam_states.erase(cam_id);
  }

  return;
}

void MsckfVio::onlineReset() {

  // Never perform online reset if position std threshold
  // is non-positive.
  if (position_std_threshold <= 0) return;
  static long long int online_reset_counter = 0;

  // Check the uncertainty of positions to determine if
  // the system can be reset.
  double position_x_std = std::sqrt(state_server.state_cov(12, 12));
  double position_y_std = std::sqrt(state_server.state_cov(13, 13));
  double position_z_std = std::sqrt(state_server.state_cov(14, 14));

  if (position_x_std < position_std_threshold &&
      position_y_std < position_std_threshold &&
      position_z_std < position_std_threshold) return;

  ROS_WARN("Start %lld online reset procedure...",
      ++online_reset_counter);
  ROS_INFO("Stardard deviation in xyz: %f, %f, %f",
      position_x_std, position_y_std, position_z_std);

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Clear all exsiting features in the map.
  map_server.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);
  
  double opti_extrinsic_rotation_cov;
  nh.param<double>("initial_covariance/opti_extrinsic_rotation_cov",
      opti_extrinsic_rotation_cov, 3.0462e-4);

  state_server.state_cov = MatrixXd::Zero(24, 24);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;
  for (int i = 21; i < 24; ++i)
    state_server.state_cov(i, i) = opti_extrinsic_rotation_cov;
  ROS_WARN("%lld online reset complete...", online_reset_counter);
  return;
}

void MsckfVio::publish(const ros::Time& time) {
  
  // add imu state to window;
  if(has_remove_state){
    if(window.size() >= 40){
      auto first_frame = window.begin();
      window.erase(first_frame);
    }
    window.push_back(state_server.imu_state);
  }

  // Convert the IMU frame to the body frame.
  const IMUState& imu_state = state_server.imu_state;
  Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
  T_i_w.linear() = quaternionToRotation(
      imu_state.orientation).transpose();
  T_i_w.translation() = imu_state.position;

  ROS_INFO("time: %f, position: %f, %f, %f", imu_state.time, imu_state.position.x(),
    imu_state.position.y(), imu_state.position.z());

  Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w *
    IMUState::T_imu_body.inverse();
  Eigen::Vector3d body_velocity =
    IMUState::T_imu_body.linear() * imu_state.velocity;

  // Publish tf;
  if (publish_tf) {
    tf::Transform T_b_w_tf;
    tf::transformEigenToTF(T_b_w, T_b_w_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_b_w_tf, time, fixed_frame_id, child_frame_id));
  }

  // Publish the odometry
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = time;
  odom_msg.header.frame_id = fixed_frame_id;
  odom_msg.child_frame_id = child_frame_id;

  tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
  tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);
  tf::vectorEigenToMsg(imu_state.opti_speed, odom_msg.twist.twist.angular);

  // Convert the covariance.
  Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
  Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
  Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
  Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
  Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
  P_imu_pose << P_pp, P_po, P_op, P_oo;

  Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
  H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
  H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
  Matrix<double, 6, 6> P_body_pose = H_pose *
    P_imu_pose * H_pose.transpose();

  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);

  // Construct the covariance for the velocity.
  Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
  Matrix3d H_vel = IMUState::T_imu_body.linear();
  Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);

  odom_pub.publish(odom_msg);

  // Publish the 3D positions of the features that
  // has been initialized.
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > feature_msg_ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  feature_msg_ptr->header.frame_id = fixed_frame_id;
  feature_msg_ptr->height = 1;
  for (const auto& item : map_server) {
    const auto& feature = item.second;
    if (feature.is_initialized) {
      Vector3d feature_position =
        IMUState::T_imu_body.linear() * feature.position;
      feature_msg_ptr->points.push_back(pcl::PointXYZ(
            feature_position(0), feature_position(1), feature_position(2)));
    }
  }
  feature_msg_ptr->width = feature_msg_ptr->points.size();

  feature_pub.publish(feature_msg_ptr);

  // gt publish
  
  Isometry3d gt = Isometry3d::Identity();
  while(gt_num < gt_poses.size()){
    if (gt_poses[gt_num].time <= state_server.imu_state.time){
        gt_num ++;
    }else{
        break;
    }
  }

  
  gt.translation() = gt_poses[gt_num].p;
  gt.linear() = gt_poses[gt_num].q.toRotationMatrix();

  state_server.cam_states[state_server.imu_state.id].imu_orientation = gt_poses[gt_num].q;

  nav_msgs::Odometry odom_gt;
  odom_gt.header.stamp = time;
  odom_gt.header.frame_id = fixed_frame_id;
  odom_gt.child_frame_id = child_frame_id;

  if(gt_num > 1){
    
    int prev_num = gt_num - 2;
    int next_num = gt_num + 2;
    double prev_time = gt_poses[prev_num].time;
    double curr_time = gt_poses[next_num].time;

    // while(curr_time - prev_time < 1e-2 && prev_num > 0){
    //   prev_num--;
    //   prev_time = gt_poses[prev_num].time;
    // }
    Eigen::Vector3d prev_p = gt_poses[prev_num].p;
    Eigen::Vector3d curr_p = gt_poses[next_num].p;
    
    Eigen::Vector3d vel = (curr_p - prev_p) / (curr_time - prev_time);

    // vel = gt_poses[gt_num].vel;
    tf::vectorEigenToMsg(vel, odom_gt.twist.twist.linear);
  }
  
  tf::poseEigenToMsg(gt, odom_gt.pose.pose);
  
  mocap_odom_pub.publish(odom_gt);

  BiasEstiInfo bias_info;
  bias_info.header.stamp = time;
  bias_info.header.frame_id = fixed_frame_id;

  bias_info.bias_acc_x = imu_state.acc_bias(0);
  bias_info.bias_acc_y = imu_state.acc_bias(1);
  bias_info.bias_acc_z = imu_state.acc_bias(2);

  bias_info.bias_gyro_x = imu_state.gyro_bias(0);
  bias_info.bias_gyro_y = imu_state.gyro_bias(1);
  bias_info.bias_gyro_z = imu_state.gyro_bias(2);

  bias_info.use_imu_num = use_imu_num;
  bias_info.opti_bias_acc_x = opti_bias_acc[0];
  bias_info.opti_bias_acc_y = opti_bias_acc[1];
  bias_info.opti_bias_acc_z = opti_bias_acc[2];
  bias_info.opti_bias_gyro_x = opti_bias_gyro[0];
  bias_info.opti_bias_gyro_y = opti_bias_gyro[1];
  bias_info.opti_bias_gyro_z = opti_bias_gyro[2];

  esti_info_pub.publish(bias_info);


  return;
}

} // namespace msckf_vio

