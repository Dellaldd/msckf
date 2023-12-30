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
        self.fold = "/home/ldd/msckf_ws/src/msckf_vio/dataset/real/data_3/filter_vel_10/"
        self.f_gt_vel = open(self.fold + "groundtruth_velocity.txt", 'w')
        self.f_filter_vel = open(self.fold + "filter_velocity.txt", 'w')
        self.f_no_filter_vel = open(self.fold + "no_filter_velocity.txt", 'w')
        
        self.gt_vel = []
        self.filter_vel = []
        self.no_filter_vel = []
        
        rospy.Subscriber("/filter_velocity", TwistStamped, self.filter_Cb)
        rospy.Subscriber("/no_filter_velocity", TwistStamped, self.no_filter_Cb)
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
        if(msg.twist.linear.z != 0.0):
            self.filter_vel.append([str(msg.header.stamp.to_sec()), str(msg.twist.linear.x), str(msg.twist.linear.y),
                str(msg.twist.linear.z)])
            
    def no_filter_Cb(self, msg):
        # msg = TwistStamped()
        if(msg.twist.linear.z != 0.0):
            self.no_filter_vel.append([str(msg.header.stamp.to_sec()), str(msg.twist.linear.x), str(msg.twist.linear.y),
                str(msg.twist.linear.z)])
            

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
                                  
    def write_title(self):        
        self.f_gt_vel.write("# timestamp vx vy vz")
        self.f_gt_vel.write('\r\n')
        
        self.f_filter_vel.write("# timestamp vx vy vz")
        self.f_filter_vel.write('\r\n')  
        
        self.f_no_filter_vel.write("# timestamp vx vy vz")
        self.f_no_filter_vel.write('\r\n')      
                
            
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


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass