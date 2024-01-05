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
        self.fold = "/home/ldd/msckf_ws/src/msckf_vio/dataset/real/data_1_4_2/"
        self.f_gt = open(self.fold + "gt.txt", 'w')
        
        self.gt = []      
        
        rospy.Subscriber("/vrpn_client_node/jiahao1/pose", PoseStamped, self.gt_Cb)
        self.add_thread = threading.Thread(target = self.thread_job)
        self.add_thread.start()

    def thread_job(self):
        rospy.spin()
        
    def write_data(self):    
        for data in self.gt:
            self.f_gt.write(' '.join(data))
            self.f_gt.write('\r\n')
                    
                          
    def write_title(self):        
        self.f_gt.write("# timestamp tx ty tz qx qy qz qw")
        self.f_gt.write('\r\n')
                
          
    def gt_Cb(self, msg):
        # msg = PoseStamped()
        t = msg.header.stamp.to_sec()
        print(t)
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w]
        print(quat)
        self.gt.append(np.array([str(t), str(pos[0]), str(pos[1]), str(pos[2]),
            str(quat[0]), str(quat[1]), str(quat[2]), str(quat[3])]))


def main():
    print("start record!")
    rospy.init_node('record_node', anonymous=True)
    logger = Logger()
    rate = rospy.Rate(200)
    logger.write_title()
    while not rospy.is_shutdown():
        rate.sleep()
    logger.write_data()
    logger.f_gt.close()
    # logger.draw()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass