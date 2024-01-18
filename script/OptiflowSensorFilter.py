import rospy
import numpy as np
from sensor_msgs.msg import Imu
from mavros_msgs.msg import VFR_HUD
from geometry_msgs.msg import TwistStamped, PoseStamped
import threading
from scipy.spatial.transform import Rotation as R
import os

class GtvelData:
    def __init__(self):
        self.time = 0
        self.velocity = np.zeros((3,1))
        
class ImuData:
    def __init__(self):
        self.time = 0
        self.acc = np.zeros((3,1))
        self.gyro = np.zeros((3,1))
        self.gyro_lpf_y = 0
        self.gyro_lpf_x = 0
        self.orien_ahrs = []

class OptiFlowData:
    def __init__(self):
        self.time = 0
        self.angular_vel_x = 0
        self.angular_vel_y = 0
        self.prev_groundspeed = 0
        self.prev_airspeed = 0
        self.prev_height_z = 0
        self.prev_vel_z = 0
        self.vel_x_out = 0
        self.vel_y_out = 0
        self.use_height = 0
        self.prev_vel_x = 0
        self.prev_vel_y = 0
                
class OptiFlowFilter:
    def __init__(self):

        # save path
        self.fold = "/home/ldd/msckf_ws/src/msckf_vio/dataset/real/data_1_18_line_1_5_1/"
        if not os.path.exists(self.fold): 
            os.mkdir(self.fold)
            
        self.f_gt = open(self.fold + "gt.txt", 'w')
        self.f_gt_vel = open(self.fold + "groundtruth_velocity.txt", 'w')
        self.f_filter_vel = open(self.fold + "filter_velocity.txt", 'w')
        self.f_no_filter_vel = open(self.fold + "no_filter_velocity.txt", 'w')
        

        self.gt = []
        self.gt_vel = []
        self.filter_vel = []
        self.no_filter_vel = []
        
        rospy.Subscriber("/mavros/imu/full", Imu, self.imuCallback)
        rospy.Subscriber("/mavros/vfr_hud", VFR_HUD, self.optiflowCallback)
        rospy.Subscriber("/outer_velocity", TwistStamped, self.gtvelCallback)
        rospy.Subscriber("/vrpn_client_node/jiahao1/pose", PoseStamped, self.gt_Cb)
        
        # publish topic
        self.filter_vel_pub = rospy.Publisher("/filter_velocity", TwistStamped, queue_size = 10)# topic 
        self.vel_pub = rospy.Publisher("/no_filter_velocity", TwistStamped, queue_size = 10)# topic 
        
        # initialize
        self.current_imu = ImuData()
        self.current_optiflow = OptiFlowData()
        self.gt_velocity = GtvelData()
        self.prev_time = 0
        self.is_first_imu = True
        self.yaw = 0
        
        self.add_thread = threading.Thread(target = self.thread_job)
        self.add_thread.start()
    
    def thread_job(self):
        rospy.spin()
    
    def gt_Cb(self, msg):
        # msg = PoseStamped()
        t = msg.header.stamp.to_sec()
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w]
        self.gt.append(np.array([str(t), str(pos[0]), str(pos[1]), str(pos[2]),
            str(quat[0]), str(quat[1]), str(quat[2]), str(quat[3])]))
    
    def gtvelCallback(self, msg):
        if(msg.twist.linear.x != 0.0):
            self.gt_vel.append([str(msg.header.stamp.to_sec()), str(msg.twist.linear.x), str(msg.twist.linear.y),
                str(msg.twist.linear.z)])

    def imuCallback(self, msg):
        
        # msg = Imu()
        self.current_imu.time = msg.header.stamp.to_sec()
        self.current_imu.acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.current_imu.gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.current_imu.orien_ahrs = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        
    def optiflowCallback(self, msg):
        flow_height = msg.altitude # mm
        self.current_optiflow.time = msg.header.stamp.to_sec()
        dt = msg.header.stamp.to_sec() - self.prev_time

        filter_vel = TwistStamped()
        no_filter_vel = TwistStamped()
        
        # remove z outlier 
        if(flow_height == self.current_optiflow.prev_height_z):
            filter_vel.twist.linear.z = self.current_optiflow.prev_vel_z 
        else:
            vz = (flow_height - self.current_optiflow.prev_height_z) / 1000 / dt / 2 / 0.5
            if(abs(vz - filter_vel.twist.linear.z) > 0.4):
                filter_vel.twist.linear.z = self.lowPassFilter(0.8, self.current_optiflow.prev_vel_z, vz)
            else:
                filter_vel.twist.linear.z = self.lowPassFilter(0.5, self.current_optiflow.prev_vel_z, vz)
            # filter_vel.twist.linear.z = self.lowPassFilter(0.5, self.current_optiflow.prev_vel_z, vz)
            self.current_optiflow.prev_height_z = flow_height
        
        # remove x outlier
        # if((abs(msg.groundspeed - self.current_optiflow.prev_groundspeed) > 15 and abs(msg.groundspeed/1000/0.02) < 0.05)
        #             or abs(msg.groundspeed/1000/0.02 - self.current_optiflow.prev_groundspeed/1000/0.02) > 1.5):
            
        # if(abs(msg.groundspeed/1000/0.02 - self.current_optiflow.prev_groundspeed/1000/0.02) > 1):
        #     groundspeed = self.current_optiflow.prev_groundspeed
        # else:
        #     groundspeed = msg.groundspeed
        # self.current_optiflow.prev_groundspeed = groundspeed
        groundspeed = msg.groundspeed
        
        # remove y outlier
        # if((abs(msg.airspeed - self.current_optiflow.prev_airspeed) > 15 and abs(msg.airspeed/1000/0.02) < 0.05)
        #             or abs(msg.airspeed/1000/0.02 - self.current_optiflow.prev_airspeed/1000/0.02) > 1.5):  
        
        # if(abs(msg.airspeed/1000/0.02 - self.current_optiflow.prev_airspeed/1000/0.02) > 1):
        #     airspeed = self.current_optiflow.prev_airspeed
        # else:
        #     airspeed = msg.airspeed
        # self.current_optiflow.prev_airspeed = airspeed
        airspeed = msg.airspeed
        
        self.current_optiflow.angular_vel_x = groundspeed / 0.02 / flow_height # rad/s
        self.current_optiflow.angular_vel_y = airspeed / 0.02 / flow_height # rad/s
        
        self.current_optiflow.use_height = flow_height
        # self.remove_outlier()
        
        if(self.prev_time != 0):
            self.fusion()
        
        # publish filter vel
        filter_vel.header = msg.header
        
        filter_vel.twist.linear.x = self.current_optiflow.vel_x_out / 1000
        filter_vel.twist.linear.y = self.current_optiflow.vel_y_out / 1000
                
        self.filter_vel_pub.publish(filter_vel)
        
        self.filter_vel.append([str(msg.header.stamp.to_sec()), str(filter_vel.twist.linear.x), str(filter_vel.twist.linear.y),
                str(filter_vel.twist.linear.z)])
          
        # publish no filter vel
        no_filter_vel.header = msg.header
        
        no_filter_vel.twist.linear.x = groundspeed/1000/0.02
        no_filter_vel.twist.linear.y = airspeed/1000/0.02
        
        # no_filter_vel.twist.linear.x = self.current_optiflow.angular_vel_x * flow_height / 1000
        # no_filter_vel.twist.linear.y = self.current_optiflow.angular_vel_y * flow_height / 1000
        
        no_filter_vel.twist.linear.z = filter_vel.twist.linear.z
        self.vel_pub.publish(no_filter_vel)
        
        self.no_filter_vel.append([str(msg.header.stamp.to_sec()), str(no_filter_vel.twist.linear.x), str(no_filter_vel.twist.linear.y),
                str(no_filter_vel.twist.linear.z)])
    
        # save prev data
        self.prev_time = msg.header.stamp.to_sec()
        self.current_optiflow.prev_height_z = flow_height
        self.current_optiflow.prev_vel_x =  filter_vel.twist.linear.x
        self.current_optiflow.prev_vel_y =  filter_vel.twist.linear.y
        self.current_optiflow.prev_vel_z = filter_vel.twist.linear.z           
            
    def lowPassFilter(self, k, in_put, out_put):
        out_put = k * in_put + (1-k) * out_put
        return out_put
    
    def limit(self, x, min, max):
        if(x > max):
            out = max
        else:
            out = x
        
        if(x < min):
            out = min
        return out
    
    def vec_3d_transition(self, acc):
        euler = R.from_quat(self.current_imu.orien_ahrs).as_euler('ZYX')
        if(self.is_first_imu):
            self.yaw = euler[0]
            self.is_first_imu = False
            
        euler[0] = euler[0] - self.yaw
        R_b_w = R.from_euler('ZYX', euler).as_matrix()
        enu_acc = np.ones((3,1))
        enu_acc[0] = acc[0] * 1000
        enu_acc[1] = - acc[1] * 1000
        enu_acc[2] = - acc[2] * 1000
        
        heading_coordinate_acc = np.dot(R_b_w, np.array(enu_acc))
        
        return heading_coordinate_acc
    
    def safe_div(self, numerator,denominator,safe_value):
        if(denominator == 0):
            return safe_value
        else:
            return numerator/denominator
    
    def filter_1(self, k, in_put, output, a):
        a = self.lowPassFilter(k,(in_put - output), a); # 低通后的变化量
        b = np.power(in_put - output, 2) # 求一个数平方函数
        e_nr = self.limit(self.safe_div(np.power(a, 2),(b + np.power(a,2)),0), 0, 1); #变化量的有效率，LIMIT 将该数限制在0-1之间，safe_div为安全除法    
        output = output + e_nr * (in_put - output) # 低通跟踪
        
        # output = self.lowPassFilter(0.8, in_put, output)
        return output, a

    def filter(self, base_hz, dT, in_put, output):
        output += ( 1 / ( 1 + 1 / ( base_hz * 3.14 * dT))) * (in_put - output)
        return output
    
    def remove_outlier(self):
        k = 0.5
        vx = self.current_optiflow.angular_vel_x * self.current_optiflow.use_height / 1000 # m/s
        vy = self.current_optiflow.angular_vel_y * self.current_optiflow.use_height / 1000
        
        print(abs( vx - self.current_optiflow.prev_vel_x))
        if(abs( vx - self.current_optiflow.prev_vel_x) > 0.3):
            self.current_optiflow.angular_vel_x = self.lowPassFilter(k, self.current_optiflow.prev_vel_x, vx) / self.current_optiflow.use_height * 1000
        
        if(abs( vy - self.current_optiflow.prev_vel_y) > 0.3):
            self.current_optiflow.angular_vel_y = self.lowPassFilter(k, self.current_optiflow.prev_vel_y, vy) / self.current_optiflow.use_height * 1000
    
    def fusion(self):
        
        dT = self.current_optiflow.time - self.prev_time
        flow_tx = 0.4
        flow_ty = 0.4
                
        self.current_imu.gyro_lpf_x = self.current_imu.gyro[0] # current gyro
        self.current_imu.gyro_lpf_y = self.current_imu.gyro[1]
                                        
        # # 光流补偿，补偿后单位为mm/s        
        fx_gyro_fix = ((self.current_optiflow.angular_vel_x  - self.limit(((self.current_imu.gyro_lpf_y)),-flow_tx,flow_tx)) * self.current_optiflow.use_height ) ;  #rotation compensation
        fy_gyro_fix = ((self.current_optiflow.angular_vel_y  - self.limit(((self.current_imu.gyro_lpf_x)),-flow_ty,flow_ty)) * self.current_optiflow.use_height ) ;  #rotation compensation
        
                       
        # 消除pitch 和 roll的影响 计算在水平平面中加速度的大小
        heading_coordinate_acc = self.vec_3d_transition(self.current_imu.acc) # 将单位转为mm/s^2
        
        # 利用加速度计测出的结果 计算当前的光流速度
        vel_x = self.current_optiflow.vel_x_out
        vel_y = self.current_optiflow.vel_y_out
        
        vel_x += float(heading_coordinate_acc[0]) * dT # 单位为mm/s
        vel_y += float(heading_coordinate_acc[1]) * dT
                        
        # # 将光流补偿后的速度和加速度计算出来的速度做互补滤波
        # k_g_x = 0.5
        # k_g_y = 0.5
        
        # vel_x, self.current_optiflow.prev_diff_x = self.filter_1(k_g_x, fx_gyro_fix, vel_x, self.current_optiflow.prev_diff_x)
        # vel_y, self.current_optiflow.prev_diff_y = self.filter_1(k_g_y, fy_gyro_fix, vel_y, self.current_optiflow.prev_diff_y)
        
        vel_x = self.filter(5, dT, fx_gyro_fix, vel_x)
        vel_y = self.filter(5, dT, fy_gyro_fix, vel_y)
        
        self.current_optiflow.vel_x_out = self.lowPassFilter(1, vel_x, self.current_optiflow.vel_x_out)
        self.current_optiflow.vel_y_out = self.lowPassFilter(1, vel_y, self.current_optiflow.vel_y_out)
        
        # self.current_optiflow.vel_x_out = vel_x
        # self.current_optiflow.vel_y_out = vel_y
            
    def write_data(self):        
        for data in self.gt_vel:
            self.f_gt_vel.write(' '.join(data))
            self.f_gt_vel.write('\r\n')
            
        for data in self.filter_vel:
            self.f_filter_vel.write(' '.join(data))
            self.f_filter_vel.write('\r\n')
        
        for data in self.no_filter_vel:
            self.f_no_filter_vel.write(' '.join(data))
            self.f_no_filter_vel.write('\r\n')
        
        for data in self.gt:
            self.f_gt.write(' '.join(data))
            self.f_gt.write('\r\n')
        
        self.f_gt.close()
        self.f_gt_vel.close()
        self.f_filter_vel.close()
        self.f_no_filter_vel.close()
        
                                  
    def write_title(self):       
        self.f_gt.write("# timestamp tx ty tz qx qy qz qw")
        self.f_gt.write('\r\n')
         
        self.f_gt_vel.write("# timestamp vx vy vz")
        self.f_gt_vel.write('\r\n')
        
        self.f_filter_vel.write("# timestamp vx vy vz")
        self.f_filter_vel.write('\r\n')  
        
        self.f_no_filter_vel.write("# timestamp vx vy vz")
        self.f_no_filter_vel.write('\r\n')                                    
              
def main():
    print("start filter!")
    rospy.init_node('filet_node', anonymous=True)
    optiflowfilter = OptiFlowFilter()
    rate = rospy.Rate(200)
    while not rospy.is_shutdown():
        rate.sleep()
    optiflowfilter.write_title()
    optiflowfilter.write_data()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass  
    
    