#!/usr/bin/env python
"""
Transform image and imu files to a ros bag file
usage  convert_files_to_bag.py [directory]
file folder:
    [color]  [depth] IMU.txt TIMESTAMP.txt
example: FMDataset
Author: Ming
"""
from __future__ import print_function
import time
import cv2
import time, sys, os
from ros import rosbag
import roslib
import rospy
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
import numpy as np
from numpy import asarray
    

def CompSortFileNamesNr(f):
    g = os.path.splitext(os.path.split(f)[1])[0] #get the file of the
    numbertext = ''.join(c for c in g if c.isdigit())
    return int(numbertext)

def ReadImages(filename):
    '''Generates a list of files from the directory'''
    print("Reading image file path from directory %s" % filename+"/TIMESTAMP.txt")
    file = open(filename+"/TIMESTAMP.txt",'r')
    all = file.readlines()
    timestamp = []
    images = []
    index = 0
    for f in all:
        index = index + 1
        if index == 1:
            continue
        line = f.rstrip('\n').split(' ')
        timestamp.append(line[0])
        images.append(line[1])
    print("Total add %i images!"%(index))
    return images,timestamp

def ReadIMU(filename):
    '''return IMU data and timestamp of IMU'''
    imu = np.loadtxt(filename + "/imu_noise.txt",delimiter=" ", skiprows=1)

    timestamp = []
    imu_data = []
    pose = []

    for i in range(imu.shape[0]):
        timestamp.append(imu[i,0])
        imu_data.append(imu[i,11:])
        pose.append(imu[i, 1:5])
    print("Total add %i imus!"%(imu.shape[0]))
    return timestamp,imu_data, pose

def ReadFeature(filename):
    '''return Feature'''
    # imu = np.loadtxt(filename + "/cam.txt",delimiter=" ", skiprows=1)
    f = open(filename + "/cam.txt")               # 返回一个文件对象 
    cam = []
    line = f.readline() 
    while line: 
        cam.append(line.split(' '))     
        line = f.readline()         
        #print(line, end = '')　      # 在 Python 3 中使用 
    f.close() 
     
    timestamp = []
    feature_data = []

    for i in range(len(cam)):
        feature = cam[i]
        timestamp.append(float(feature[0]))
        feature_data.append(feature[1:])
    print("Total add %i cams!"%(len(cam)))
    return timestamp,feature_data


def CreateBag(foldpath):
    # imu
    imutimesteps,imudata, pose = ReadIMU(foldpath) 
    cameratimesteps,cameradata = ReadFeature(foldpath) 
    
    bag = rosbag.Bag(foldpath + "data_noise.bag", 'w')

    try:
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
            orien.w = float(pose[i][0])
            orien.x = float(pose[i][1])
            orien.y = float(pose[i][2])
            orien.z = float(pose[i][3])
            
            imuStamp = rospy.rostime.Time.from_sec(float(imutimesteps[i]))
            imu.header.stamp = imuStamp
            imu.angular_velocity = angular_v
            imu.linear_acceleration = linear_a
            imu.orientation = orien
            
            bag.write("/imu",imu,imuStamp)
        
        for i in range(len(cameradata)):   
            camera_msg = CameraMeasurement()
            
            cameraStamp = rospy.rostime.Time.from_sec(float(cameratimesteps[i]))
            
            camera_msg.header.stamp = cameraStamp
            camera = cameradata[i]
            
            n = (len(camera))/5
         
            
            for id in range(int(n)):
                feature = FeatureMeasurement()
                feature.id = int(camera[int(id*5)])
                feature.u0 = float(camera[int(id*5+1)])
                feature.v0 = float(camera[int(id*5+2)])
                feature.u1 = float(camera[int(id*5+3)])
                feature.v1 = float(camera[int(id*5+4)])
                camera_msg.features.append(feature)
            bag.write("/features", camera_msg, cameraStamp)
                
            
    finally:
        bag.close()

if __name__ == "__main__":
    foldpath = "/home/ldd/bias_esti_ws/src/bias_esti/datasets/imu_simulation/feature_noise/test_model/big_noise_feature_10/"
    print(foldpath)
    CreateBag(foldpath)