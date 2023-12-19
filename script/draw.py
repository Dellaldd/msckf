import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from tf.transformations import euler_from_quaternion
import math


def main():
    
    foldpath = "/home/ldd/msckf_ws/src/msckf_vio/dataset/real/data_1/"
    gt_path = foldpath + "groundtruth_velocity.txt"
    opti_path = foldpath + "traj_estimate_velocity.txt"
    gt_vel = np.loadtxt(gt_path, delimiter=' ', skiprows=1)
    opti_vel = np.loadtxt(opti_path, delimiter=' ', skiprows=1)
    
    fig1, ax1 = plt.subplots(1, 3)
    gt_vel = np.array(gt_vel)
    opti_vel = np.array(opti_vel)
    
    ax1[0].plot(gt_vel[:,0], gt_vel[:,1], 'r-', label = 'gt')
    ax1[1].plot(gt_vel[:,0], gt_vel[:,2], 'r-', label = 'gt')
    ax1[2].plot(gt_vel[:,0], gt_vel[:,3], 'r-', label = 'gt')
    
    ax1[0].plot(opti_vel[:,0], opti_vel[:,1], 'b-', label = 'opti')
    ax1[1].plot(opti_vel[:,0], opti_vel[:,2], 'b-', label = 'opti')
    ax1[2].plot(opti_vel[:,0], opti_vel[:,3]/1000, 'b-', label = 'opti')
    
    lines, labels = fig1.axes[-1].get_legend_handles_labels()
    fig1.legend(lines, labels, loc = 'upper right') # 图例的位置
    fig1.tight_layout()
    
    save_path = foldpath + "result.png"
    plt.savefig(save_path, dpi=300)

    plt.show()
    

if __name__ == '__main__':

    main()
    
