import rospy
import numpy as np
from sensor_msgs.msg import Imu
from mavros_msgs.msg import VFR_HUD
from geometry_msgs.msg import TwistStamped
import threading

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

class OptiFlowData:
    def __init__(self):
        self.time = 0
        self.fx = 0
        self.fy = 0
        self.prev_fx = 0
        self.prev_fy = 0
        self.f1_fx_out = 0
        self.f1_fy_out = 0
        self.use_height = 0
        self.a_x = 0
        self.a_y = 0
        self.error_x = 0
        self.error_y = 0
        self.f_out_x = 0
        self.f_out_y = 0
        
class OptiFlowFilter:
    def __init__(self):
        rospy.Subscriber("/mavros/imu/full", Imu, self.imuCallback)
        rospy.Subscriber("/mavros/vfr_hud", VFR_HUD, self.optiflowCallback)
        # rospy.Subscriber("/outer_velocity", TwistStamped, self.gtvelCallback)
        self.filter_vel_pub = rospy.Publisher("/filter_velocity", TwistStamped, queue_size = 10)# topic 
        self.vel_pub = rospy.Publisher("/no_filter_velocity", TwistStamped, queue_size = 10)# topic 
        
        self.current_imu = ImuData()
        self.current_optiflow = OptiFlowData()
        self.gt_velocity = GtvelData()
        self.prev_time = 0
        
        self.add_thread = threading.Thread(target = self.thread_job)
        self.add_thread.start()
    
    def thread_job(self):
        rospy.spin()
        
    def imuCallback(self, msg):
        # msg = Imu()
        self.current_imu.time = msg.header.stamp.to_sec()
        self.current_imu.acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.current_imu.gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        
    def optiflowCallback(self, msg):
        # msg = VFR_HUD()
        flow_height = msg.climb/10
        self.current_optiflow.time = msg.header.stamp.to_sec()
        self.current_optiflow.fx = msg.groundspeed / 0.02 / 10 / flow_height
        self.current_optiflow.fy = msg.airspeed / 0.02 / 10 / flow_height
        
        self.current_optiflow.use_height = flow_height
        
        # if(flow_height<200):    # input height (cm/s) ?
        #     self.current_optiflow.use_height = 100
        # elif(flow_height<300):
        #     self.current_optiflow.use_height = 150
        # elif(flow_height<400):
        #     self.current_optiflow.use_height = 200
        # elif(flow_height<500):
        #     self.current_optiflow.use_height = 250
        # elif(flow_height<600):
        #     self.current_optiflow.use_height = 300
        # elif(flow_height<1000):
        #     self.current_optiflow.use_height = 350
        # else: 
        #     self.current_optiflow.use_height = 400
        
        if(self.prev_time != 0):
            self.fusion()
            
        self.prev_time = msg.header.stamp.to_sec()
        self.current_optiflow.prev_fx = self.current_optiflow.fx
        self.current_optiflow.prev_fy = self.current_optiflow.fy
        
        filter_vel = TwistStamped()
        filter_vel.header = msg.header
        filter_vel.twist.linear.x = self.current_optiflow.f_out_x / 1000
        filter_vel.twist.linear.y = self.current_optiflow.f_out_y / 1000
        self.filter_vel_pub.publish(filter_vel)
        
        no_filter_vel = TwistStamped()
        no_filter_vel.header = msg.header
        no_filter_vel.twist.linear.x = msg.groundspeed/1000/0.02
        no_filter_vel.twist.linear.y = msg.airspeed/1000/0.02
        self.vel_pub.publish(no_filter_vel)
    
    # def gtvelCallback(self, msg):
    #     # msg = TwistStamped()
    #     self.gt_velocity.time = msg.header.stamp.to_sec()
    #     self.gt_velocity.velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
    
    def lowPassFilter(self, hz, t, in_put, out_put):
        out_put += ( 1 / ( 1 + 1 / ( hz * 3.14 * t))) * (in_put - out_put)
        return out_put
    
    def limit(self, x, min, max):
        if(x > max):
            out = max
        else:
            out = x
        
        if(x < min):
            out = min
        return out
        # ((x) < (min)) ? (min) : ( ((x) > (max))? (max) : (x) )
    
    def vec_3d_transition(self, acc):
        heading_coordinate_acc = acc
        heading_coordinate_acc[0] = heading_coordinate_acc[0] * 1000
        heading_coordinate_acc[1] = - heading_coordinate_acc[1] * 1000
        heading_coordinate_acc[2] = - heading_coordinate_acc[2] * 1000
        return heading_coordinate_acc
    
    def safe_div(self, numerator,denominator,safe_value):
        if(denominator == 0):
            return safe_value
        else:
            return numerator/denominator
        # ( (denominator == 0)? (safe_value) : ((numerator)/(denominator)) )
    
    def filter_1(self, base_hz, gain_hz, dT, in_put, output, a):
        a = self.lowPassFilter(gain_hz, dT, (in_put - output), a); # 低通后的变化量
        b = np.power(in_put - output, 2) # 求一个数平方函数
        e_nr = self.limit(self.safe_div(np.power(a, 2),(b + np.power(a,2)),0), 0, 1); #变化量的有效率，LIMIT 将该数限制在0-1之间，safe_div为安全除法
        
        output = self.lowPassFilter(base_hz *e_nr, dT, in_put, output) # 低通跟踪
        return output, a
    
    def pi_control(self, f_acc, f_opti, kp):
        f = f_acc + kp * (f_opti - f_acc)
        return f
      
    def fusion(self):
        
        
        # filter
        dT = self.current_optiflow.time - self.prev_time
        flow_t1 = 1
                
        self.current_imu.gyro_lpf_x = self.current_imu.gyro[0]
        self.current_imu.gyro_lpf_y = self.current_imu.gyro[1]
        
        # self.current_imu.gyro_lpf_x = self.lowPassFilter(8.0, dT, self.current_imu.gyro[0], self.current_imu.gyro_lpf_x)
        # self.current_imu.gyro_lpf_y = self.lowPassFilter(8.0, dT, self.current_imu.gyro[1], self.current_imu.gyro_lpf_y)
          
        self.current_optiflow.fx = self.lowPassFilter(30.0, dT, self.current_optiflow.fx, self.current_optiflow.prev_fx)
        self.current_optiflow.fy = self.lowPassFilter(30.0, dT, self.current_optiflow.fy, self.current_optiflow.prev_fy)
                
        # # # 光流补偿，补偿后单位为mm/s        
        fx_gyro_fix = ((self.current_optiflow.fx  + self.limit(((self.current_imu.gyro_lpf_x)),-flow_t1,flow_t1)) * 10 * self.current_optiflow.use_height ) ;  #rotation compensation
        fy_gyro_fix = ((self.current_optiflow.fy  + self.limit(((-self.current_imu.gyro_lpf_y)),-flow_t1,flow_t1)) * 10 * self.current_optiflow.use_height ) ;  #rotation compensation
        
        
        # # 消除pitch 和 roll的影响 计算在水平平面中加速度的大小
        heading_coordinate_acc = self.vec_3d_transition(self.current_imu.acc) # 将单位转为mm/s^2
        
        # # 利用加速度计测出的结果 计算当前的光流速度
        self.current_optiflow.f1_fx_out += heading_coordinate_acc[0] * dT # 单位为mm/s
        self.current_optiflow.f1_fy_out += heading_coordinate_acc[1] * dT
                
        # # 将光流补偿后的速度和加速度计算出来的速度做互补滤波
        f1_b = 10
        f1_g = 2.5
        self.current_optiflow.f1_fx_out, self.current_optiflow.a_x = self.filter_1(f1_b, f1_g, dT, fx_gyro_fix, self.current_optiflow.f1_fx_out, self.current_optiflow.a_x)
        self.current_optiflow.f1_fy_out, self.current_optiflow.a_y = self.filter_1(f1_b, f1_g, dT, fy_gyro_fix, self.current_optiflow.f1_fy_out, self.current_optiflow.a_y)
        
        self.current_optiflow.f_out_x = self.current_optiflow.f1_fx_out
        self.current_optiflow.f_out_y = self.current_optiflow.f1_fy_out
                       
        # # 融合速度二次修正，最终输出结果
        self.current_optiflow.f_out_x = 0.9 * (self.current_optiflow.f1_fx_out + 0.1 * self.current_optiflow.error_x) + 0.1 * self.current_optiflow.f_out_x
        self.current_optiflow.f_out_y = 0.9 * (self.current_optiflow.f1_fy_out + 0.1 * self.current_optiflow.error_y) + 0.1 * self.current_optiflow.f_out_y
        
        self.current_optiflow.error_x += (fx_gyro_fix - self.current_optiflow.f_out_x) * dT
        self.current_optiflow.error_y += (fy_gyro_fix - self.current_optiflow.f_out_y) * dT
        
        print("x: ", self.current_optiflow.f_out_x/1000, fx_gyro_fix/1000)
        print("y: ", self.current_optiflow.f_out_y/1000, fy_gyro_fix/1000)
        
        # 这里为什么不需要将修正后的结果赋值给f_out_x？ 下一次计算当前的光流速度用修正后的速度？
        self.current_optiflow.f1_fx_out = self.current_optiflow.f_out_x
        self.current_optiflow.f1_fy_out = self.current_optiflow.f_out_y
        
        #  use pi control
        # kp_x = 0.01
        # kp_y = 0.01
        # self.current_optiflow.f1_fx_out = self.pi_control(self.current_optiflow.f1_fx_out, fx_gyro_fix, kp_x)
        # self.current_optiflow.f1_fy_out = self.pi_control(self.current_optiflow.f1_fy_out, fy_gyro_fix, kp_y)
        # self.current_optiflow.f_out_x = self.current_optiflow.f1_fx_out
        # self.current_optiflow.f_out_y = self.current_optiflow.f1_fy_out
        
              
def main():
    print("start filter!")
    rospy.init_node('filet_node', anonymous=True)
    optiflowfilter = OptiFlowFilter()
    rate = rospy.Rate(200)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass  
    
    