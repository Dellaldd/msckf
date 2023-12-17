from __future__ import print_function
import time
import cv2
import time, sys, os
from ros import rosbag
import roslib
import rospy
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion, Vector3Stamped
import numpy as np
from PIL import Image as ImagePIL
from cv_bridge import CvBridge
import random

def ReadImages(filename):
    path = filename + "/cam1/data.txt"
    
    fileHandler = open(path,"r")
    listOfLines = fileHandler.readlines()
    fileHandler.close()
    timestamp = []
    count = 0
    for line in listOfLines:
        if(count == 0):
            count += 1
            continue
        l = line.split(',')
        timestamp.append(l[0])
 
    image0_path = filename + "/cam0/data/"
    image1_path = filename + "/cam1/data/"
       
    data0 = []
    data1 = []
    
    for i in range(len(timestamp)):
        path0 = image0_path + str(timestamp[i]) + ".png"
        data0.append(path0)
        
        path1 = image1_path + str(timestamp[i]) + ".png"
        data1.append(path1)
        
    print("Total add %i cam0s %i cam0s %i!"%(len(data0), len(data1), len(timestamp)))
    
    return timestamp, data0, data1

def ReadSpeed(filename):
    imu = np.loadtxt(filename + "/gt.txt",delimiter=",", skiprows=1)
    timestamp = []
    data = []

    for i in range(imu.shape[0]):
        timestamp.append(imu[i,0])
        data.append(imu[i,8:11])
    print("Total add %i imus!"%(imu.shape[0]))
    return timestamp, data

def ReadIMU(filename):
    imu = np.loadtxt(filename + "/imu.txt",delimiter=",", skiprows=1)
    timestamp = []
    imu_data = []

    for i in range(imu.shape[0]):
        timestamp.append(imu[i,0])
        imu_data.append(imu[i,1:])
    print("Total add %i imus!"%(imu.shape[0]))
    return timestamp, imu_data

def CreateBag(foldpath, noise, deviation, bag_save_path):
    # imu
    imutimesteps, imudata = ReadIMU(foldpath) 
    speedtimesteps, speeddata = ReadSpeed(foldpath)
    cameratimesteps, cam0, cam1 = ReadImages(foldpath) 
       
    bag = rosbag.Bag(bag_save_path + "data.bag", 'w')
    f_opti_speed = open(bag_save_path + "/opti_speed.txt", 'w')
    
    br = CvBridge()

    for i in range(len(imudata)):
        imu = Imu()
        angular_v = Vector3()
        linear_a = Vector3()
        orien = Quaternion()
        angular_v.x = float(imudata[i][0])
        angular_v.y = float(imudata[i][1])
        angular_v.z = float(imudata[i][2])
        linear_a.x = float(imudata[i][3])
        linear_a.y = float(imudata[i][4])
        linear_a.z = float(imudata[i][5])
        
        imuStamp = rospy.rostime.Time.from_sec(float(imutimesteps[i] * 1e-9))
        imu.header.stamp = imuStamp
        imu.angular_velocity = angular_v
        imu.linear_acceleration = linear_a
        imu.orientation = orien
        
        bag.write("/imu",imu,imuStamp)
    
    opti_speed = []
    for i in range(len(speedtimesteps)):
        speed = Vector3Stamped()           
        speedStamp = rospy.rostime.Time.from_sec(float(speedtimesteps[i] * 1e-9))
        speed.header.stamp = speedStamp
        if( not noise):
            speed.vector.x = float(speeddata[i][0] + random.gauss(0, deviation) * noise)
            speed.vector.y = float(speeddata[i][1] + random.gauss(0, deviation) * noise)
            speed.vector.z = float(speeddata[i][2] + random.gauss(0, deviation) * noise)
        else:
            speed.vector.x = float(speeddata[i][0])
            speed.vector.y = float(speeddata[i][1])
            speed.vector.z = float(speeddata[i][2])
        
        opti_speed.append(np.array([speedtimesteps[i], speed.vector.x, speed.vector.y, speed.vector.z]))
        bag.write("/speed", speed, speedStamp)
    
    for i in range(len(cameratimesteps)):  
        print(i)
        valid = os.path.exists(cam0[i]) 
        if(valid):  
            img0 = ImagePIL.open(cam0[i])
            img0 = np.asarray(img0)
            img0_msg = br.cv2_to_imgmsg(img0)  # Convert the color image to a message
            camstamp = rospy.rostime.Time.from_sec(float(cameratimesteps[i]) * 1e-9)
            img0_msg.header.stamp = camstamp
            
            img0_msg.header.frame_id = "camera"
            img0_msg.encoding = "mono8"
            bag.write('cam0', img0_msg, camstamp)
        
        valid = os.path.exists(cam1[i])
        if(valid):
            img1 = ImagePIL.open(cam1[i])
            img1 = np.asarray(img1)
            img1_msg = br.cv2_to_imgmsg(img1)  # Convert the color image to a message
            camstamp = rospy.rostime.Time.from_sec(float(cameratimesteps[i]) * 1e-9)
            img1_msg.header.stamp = camstamp
            
            img1_msg.header.frame_id = "camera"
            img1_msg.encoding = "mono8"
            bag.write('/cam1', img1_msg, camstamp)

    # write opti speed to txt
    f_opti_speed.write('time x y z')
    f_opti_speed.write('\r\n')
    for id in range(len(opti_speed)):
        data = [str(opti_speed[id][0]), str(opti_speed[id][1]), str(opti_speed[id][2]), str(opti_speed[id][3])]
        f_opti_speed.write(' '.join(data))
        f_opti_speed.write('\r\n')
    f_opti_speed.close()

if __name__ == "__main__":
    foldpath = "/home/ldd/msckf_ws/src/msckf_vio/dataset/v203/"
    bag_save_path = foldpath + "noise_0_1/"
    noise = 0.1
    deviation = 1
    print(foldpath)
    CreateBag(foldpath, noise, deviation, bag_save_path)