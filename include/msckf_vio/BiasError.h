#ifndef BiasError_H
#define BiasError_H

#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <msckf_vio/imu_state.h>

using namespace std;
using namespace Eigen;

namespace msckf_vio{
class BiasError {
public:
    BiasError(Eigen::Vector3d gyro_bias, double prev_time, Eigen::Vector3d prev_acc, Eigen::Vector3d prev_gyro, Eigen::Vector3d speed_i, Eigen::Vector3d speed_j, std::vector<std::pair<double, Eigen::Vector3d>> acc_set,
        std::vector<std::pair<double, Eigen::Vector3d>> gyro_set, Eigen::Quaterniond orien):
        gyro_bias(gyro_bias), prev_time(prev_time), prev_acc(prev_acc), prev_gyro(prev_gyro), speed_i(speed_i), speed_j(speed_j), acc_set(acc_set), gyro_set(gyro_set), orien(orien){}
                                                                           
    template<typename T>
    bool operator()(const T *const ba, const T *const bw, T *residuals) const {
        
        Eigen::Matrix<T, 3, 1> bias_a(ba[0], ba[1], ba[2]);
        Eigen::Matrix<T, 3, 1> bias_w(bw[0], bw[1], bw[2]);
        Eigen::Quaternion<T> q = orien.cast<T>(); // R_i_w
        Eigen::Matrix<T, 3, 1> vel = speed_i.cast<T>();
        Eigen::Matrix<T, 3, 1> opti_speed = speed_j.cast<T>();

        Eigen::Matrix<T, 3, 1> prev_a, prev_w;
        Eigen::Matrix<T, 3, 1> G = IMUState::gravity.cast<T>();  

        for(int i = 0 ; i < acc_set.size(); i++){
            T dt;
            if(i == 0){
                dt = T(acc_set[i].first - prev_time);
                prev_a = prev_acc.cast<T>();
                prev_w = prev_gyro.cast<T>();
            }else{
                dt = T(acc_set[i].first - acc_set[i-1].first);
                prev_a = acc_set[i-1].second.cast<T>();
                prev_w = gyro_set[i-1].second.cast<T>();
            }
            
            Eigen::Matrix<T, 3, 1> acc_no_bias0 = prev_a - bias_a;
            Eigen::Matrix<T, 3, 1> gyro_no_bias0 = prev_w - bias_w;
            Eigen::Matrix<T, 3, 1> acc_no_bias = acc_set[i].second.cast<T>() - bias_a;
            Eigen::Matrix<T, 3, 1> gyro_no_bias = gyro_set[i].second.cast<T>() - bias_w;

            Eigen::Matrix<T, 3, 1> un_acc_g_0 = q * acc_no_bias0 + G;
            Eigen::Matrix<T, 3, 1> un_gyr = T(0.5) * (gyro_no_bias0 + gyro_no_bias) ;
            // q = q * Eigen::Quaternion<T>(T(1), un_gyr(0) * dt / T(2), un_gyr(1) * dt / T(2), un_gyr(2) * dt / T(2));
            // q.normalize();
            Eigen::Matrix<T, 3, 1> un_acc_g_1 = q * acc_no_bias + G; 
            Eigen::Matrix<T, 3, 1> un_acc_g = T(0.5) * (un_acc_g_0 + un_acc_g_1);

            vel += dt * un_acc_g;
        }
        
        residuals[0] = ceres::abs(opti_speed[0] - vel[0]);
        residuals[1] = ceres::abs(opti_speed[1] - vel[1]);
        residuals[2] = ceres::abs(opti_speed[2] - vel[2]);

        residuals[3] = T(10) * ceres::abs(bw[0] - T(gyro_bias[0]));
        residuals[4] = T(10) * ceres::abs(bw[1] - T(gyro_bias[1]));
        residuals[5] = T(10) * ceres::abs(bw[2] - T(gyro_bias[2]));
              
        return true;
    }

    


private:
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d speed_i, speed_j, prev_acc, prev_gyro; 
  std::vector<std::pair<double, Eigen::Vector3d>> acc_set, gyro_set;
  Eigen::Quaterniond orien;
  double prev_time;
};


}
#endif // PreIntegrateReprojectError.h