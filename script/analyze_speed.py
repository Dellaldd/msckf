#!/usr/bin/env python
import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt

import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import threading

class Logger:
    def __init__(self):
        self.fold = "/home/ldd/msckf_ws/src/msckf_vio/dataset/real/data_12_22_2/"
        self.f_gt_vel = open(self.fold + "groundtruth_velocity.txt", 'w')
        self.f_vel = open(self.fold + "traj_estimate_velocity.txt", 'w')
        
        self.gt_vel = []
        self.opti_vel = []
        self.prev_gt_time = 0
        self.prev_gt_pos = [0, 0, 0]
        
        self.prev_time = 0
        self.prev_z = 0
        
        rospy.Subscriber("/vrpn_client_node/jiahao1/pose", PoseStamped, self.gt_Cb)
        rospy.Subscriber("/mavros/vfr_hud", VFR_HUD, self.opti_Cb)
        # rospy.Subscriber("/outer_velocity", TwistStamped, self.gt_Cb)
        self.add_thread = threading.Thread(target = self.thread_job)
        self.add_thread.start()

    def thread_job(self):
        rospy.spin()
        
    def write_data(self):    
        print(len(self.gt_vel), len(self.opti_vel))
        for data in self.gt_vel:
            self.f_gt_vel.write(' '.join(data))
            self.f_gt_vel.write('\r\n')
        
        for data in self.opti_vel:
            self.f_vel.write(' '.join(data))
            self.f_vel.write('\r\n')
            
                          
    def write_title(self):        
        self.f_gt_vel.write("# timestamp vx vy vz")
        self.f_gt_vel.write('\r\n')
        
        self.f_vel.write("# timestamp vx vy vz")
        self.f_vel.write('\r\n')
        
          
    def gt_Cb(self, msg):
        if(self.prev_gt_time != 0):
            t = msg.header.stamp.to_sec()
            pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            vel = (pos - np.array(self.prev_gt_pos))/(t - self.prev_gt_time)
            if(vel[0] != 0):
                self.gt_vel.append(np.array([str(t), str(vel[0]), str(vel[1]), str(vel[2])]))
                print("pos: ", pos)
                print("gt vel: ", vel, "time: ", t - self.prev_gt_time)      

            
        self.prev_gt_time = msg.header.stamp.to_sec()
        self.prev_gt_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z] 
        
        # if(msg.twist.linear.x != 0):
        #     t = msg.header.stamp.to_sec()
        #     vel = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        #     self.gt_vel.append(np.array([str(t), str(vel[0]), str(vel[1]), str(vel[2])]))


    def opti_Cb(self, msg):
        if(self.prev_time != 0):
            t = msg.header.stamp.to_sec()
            p_z = msg.climb
            vel = (p_z - self.prev_z)/(t - self.prev_time)
            self.opti_vel.append(np.array([str(t), str(msg.airspeed), str(msg.groundspeed), str(vel)]))
            print("vel: ", vel)
        self.prev_time = msg.header.stamp.to_sec()
        self.prev_p = msg.climb  
    
    def draw(self):
        fig1, ax1 = plt.subplots(1, 3)
        self.gt_vel = np.array(self.gt_vel)
        self.opti_vel = np.array(self.opti_vel)
        
        ax1[0].plot(self.gt_vel[:,0], self.gt_vel[:,1], 'r-', label = 'gt')
        ax1[1].plot(self.gt_vel[:,0], self.gt_vel[:,2], 'r-', label = 'gt')
        ax1[2].plot(self.gt_vel[:,0], self.gt_vel[:,3], 'r-', label = 'gt')
        
        ax1[0].plot(self.opti_vel[:,0], self.opti_vel[:,1], 'r-', label = 'opti')
        ax1[1].plot(self.opti_vel[:,0], self.opti_vel[:,2], 'r-', label = 'opti')
        ax1[2].plot(self.opti_vel[:,0], self.opti_vel[:,3], 'r-', label = 'opti')
        
        lines, labels = fig1.axes[-1].get_legend_handles_labels()
        fig1.legend(lines, labels, loc = 'upper right') # 图例的位置
        fig1.tight_layout()
        
        save_path = self.fold + "result.png"
        plt.savefig(save_path, dpi=300)
    
        plt.show()
def main():
    print("start record!")
    rospy.init_node('record_node', anonymous=True)
    logger = Logger()
    rate = rospy.Rate(200)
    logger.write_title()
    while not rospy.is_shutdown():
        rate.sleep()
    logger.write_data()
    logger.f_vel.close()
    logger.f_gt_vel.close()
    # logger.draw()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass