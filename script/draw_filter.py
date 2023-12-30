import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from tf.transformations import euler_from_quaternion
import math


def main():
    
    foldpath = "/home/ldd/msckf_ws/src/msckf_vio/dataset/real/data_4/filter_vel_10/"
    gt_path = foldpath + "groundtruth_velocity.txt"
    opti_path = foldpath + "filter_velocity.txt"
    no_filter_path = foldpath + "no_filter_velocity.txt"
    
    gt_vel = np.loadtxt(gt_path, delimiter=' ', skiprows=1)
    opti_vel = np.loadtxt(opti_path, delimiter=' ', skiprows=1)
    no_filter = np.loadtxt(no_filter_path, delimiter=' ', skiprows=1)
    
    fig1, ax1 = plt.subplots(2, 3)
    gt_vel = np.array(gt_vel)
    opti_vel = np.array(opti_vel)
    
    ax1[0, 0].plot(gt_vel[:,0], gt_vel[:,1], 'r-', label = 'gt')
    ax1[0, 1].plot(gt_vel[:,0], gt_vel[:,2], 'r-', label = 'gt')
    ax1[0, 2].plot(gt_vel[:,0], gt_vel[:,3], 'r-', label = 'gt')
    
    ax1[0, 0].plot(opti_vel[:,0], opti_vel[:,1], 'b-', label = 'opti')
    ax1[0, 1].plot(opti_vel[:,0], opti_vel[:,2], 'b-', label = 'opti')
    # ax1[2].plot(opti_vel[:,0], opti_vel[:,3], 'b-', label = 'opti')
    
    ax1[0, 0].plot(no_filter[:,0], no_filter[:,1], 'g-', label = 'no_filter')    
    ax1[0, 1].plot(no_filter[:,0], no_filter[:,2], 'g-', label = 'no_filter')
    ax1[0, 2].plot(no_filter[:,0], no_filter[:,3], 'g-', label = 'no_filter')
    
    # error = []
    # for i in range(1,4):
    #     error_x = np.array(gt_vel[:,i] - no_filter[:,i])
    #     error_x = np.multiply(error_x, error_x)
    #     error_x = np.sum(error_x, axis=1)
    #     error.append(np.sum(np.sqrt(error_x))/error_x.shape[0])
    
    # print(error)
    
    
    lines, labels = fig1.axes[-1].get_legend_handles_labels()
    fig1.legend(lines, labels, loc = 'upper right') # 图例的位置
    fig1.tight_layout()
    
    save_path = foldpath + "result1.png"
    plt.savefig(save_path, dpi=300)

    plt.show()
    

if __name__ == '__main__':

    main()
    
