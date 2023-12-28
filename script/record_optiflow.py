#!/usr/bin/env python
import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from mavros_msgs.srv import *

from scipy.spatial.transform import Rotation as R
import numpy as np
import threading
import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.fold = "/home/ldd/msckf_ws/src/msckf_vio/dataset/real/data_1/filter/"
        self.f_gt_vel = open(self.fold + "groundtruth_velocity.txt", 'w')
        self.f_filter_vel = open(self.fold + "filter_velocity.txt", 'w')
        
        self.gt_vel = []
        self.filter_vel = []
        
        rospy.Subscriber("/filter_velocity", TwistStamped, self.filter_Cb)
        rospy.Subscriber("/outer_velocity", TwistStamped, self.gt_Cb)

        self.add_thread = threading.Thread(target = self.thread_job)
        self.add_thread.start()

    def thread_job(self):
        rospy.spin()
    
    def gt_Cb(self, msg):
        if(msg.twist.linear.x != 0.0):
            self.gt_vel.append([str(msg.header.stamp.to_sec()), str(msg.twist.linear.x), str(msg.twist.linear.y),
                str(msg.twist.linear.z)])
    
    def filter_Cb(self, msg):
        # msg = TwistStamped()
        if(msg.twist.linear.x != 0.0):
            self.filter_vel.append([str(msg.header.stamp.to_sec()), str(msg.twist.linear.x), str(msg.twist.linear.y),
                str(msg.twist.linear.z)])

    def write_data(self):        
        for data in self.gt_vel:
            self.f_gt_vel.write(' '.join(data))
            self.f_gt_vel.write('\r\n')
            
        for data in self.filter_vel:
            self.f_filter_vel.write(' '.join(data))
            self.f_filter_vel.write('\r\n')
                                  
    def write_title(self):        
        self.f_gt_vel.write("# timestamp vx vy vz")
        self.f_gt_vel.write('\r\n')
        self.f_filter_vel.write("# timestamp vx vy vz")
        self.f_filter_vel.write('\r\n')        
    
    def draw(self):
        fig1, ax1 = plt.subplots(1, 3)
        self.gt_vel = np.array(self.gt_vel)
        self.filter_vel = np.array(self.filter_vel)
        
        ax1[0].plot(self.gt_vel[:,0], self.gt_vel[:,1], 'r-', label = 'gt')
        ax1[1].plot(self.gt_vel[:,0], self.gt_vel[:,2], 'r-', label = 'gt')
        ax1[2].plot(self.gt_vel[:,0], self.gt_vel[:,3], 'r-', label = 'gt')
        
        ax1[0].plot(self.filter_vel[:,0], self.filter_vel[:,1], 'b-', label = 'filter')
        ax1[1].plot(self.filter_vel[:,0], self.filter_vel[:,2], 'b-', label = 'filter')
        ax1[2].plot(self.filter_vel[:,0], self.filter_vel[:,3], 'b-', label = 'filter')
        
        lines, labels = fig1.axes[-1].get_legend_handles_labels()
        fig1.legend(lines, labels, loc = 'upper right') # 图例的位置
        fig1.tight_layout()
        
        save_path = self.fold + "result.png"
        plt.savefig(save_path, dpi=300)
    
        # plt.show()
            
            
def main():
    print("start record!")
    rospy.init_node('record_node', anonymous=True)
    logger = Logger()
    rate = rospy.Rate(200)
    logger.write_title()
    while not rospy.is_shutdown():
        rate.sleep()
    logger.write_data()
    logger.f_gt_vel.close()
    logger.f_filter_vel.close()
    # logger.draw()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass