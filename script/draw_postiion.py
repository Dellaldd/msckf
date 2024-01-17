import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

def quaternion_to_euler(q, degree_mode=1):
    qw, qx, qy, qz = q

    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    pitch = math.asin(2 * (qw * qy - qz * qx))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    # degree_mode=1:【输出】是角度制，否则弧度制
    if degree_mode == 1:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)
    euler = np.array([roll, pitch, yaw])
    return euler


def main():
    foldpath = "/home/ldd/msckf_ws/src/msckf_vio/result/msckf/real/data_1_16_line_1/"
    msckf_foldpath = "/home/ldd/msckf_ws/src/msckf_vio/result/msckf/v203/"
    
    gt_path = foldpath + "groundtruth_velocity.txt"  
    esti_path = foldpath + "traj_estimate_velocity.txt"
    bias_path = foldpath + "bias.txt"
    new_esti_path = foldpath + "stamped_traj_estimate.txt"
    msckf_bias_path = msckf_foldpath + "bias.txt"
    is_euroc = True
    use_msckf = False
    
    save_position_path = foldpath + "position.png"
    save_bias_path = foldpath + "bias.png"
    
    gt = np.loadtxt(gt_path, delimiter=' ', skiprows=1)
    esti = np.loadtxt(esti_path, delimiter=' ', skiprows=1)
    bias = np.loadtxt(bias_path, delimiter=' ', skiprows=1)
    msckf_bias = np.loadtxt(msckf_bias_path, delimiter=' ', skiprows=1)
    f_new_esti = open(new_esti_path, 'w')
    
    new_esti_state = []
    
    # print(gt.shape)
    eulers = []
    eulers_gt = []
    error_pos = []
    
    end = min(esti.shape[0], gt.shape[0])
    start = 0
    
    for i in range(end):
        q = [esti[i,4], esti[i,5], esti[i,6], esti[i,7]]
        euler = list(euler_from_quaternion(q))
        
        if(is_euroc):
            if euler[0] > 0:
                euler[0] -= np.pi
            else:
                euler[0] += np.pi
                
        eulers.append(euler)
        new_q = quaternion_from_euler(euler[0], euler[1], euler[2])
        
        new_esti_state.append([str(esti[i,0]), str(esti[i,1]), str(esti[i,2]), str(esti[i,3]),
            str(new_q[0]), str(new_q[1]), str(new_q[2]), str(new_q[3])])
        
        # gt
        q_gt = [gt[i,4], gt[i,5], gt[i,6], gt[i,7]]
        euler_gt = list(euler_from_quaternion(q_gt))
        
        # if(is_euroc):
        #     if euler_gt[0] > 0:
        #         euler_gt[0] -= np.pi
        #     else:
        #         euler_gt[0] += np.pi
        
        eulers_gt.append(euler_gt)
        
        error_pos.append(np.array([gt[i,1]-esti[i,1], gt[i,2]-esti[i,2], gt[i,3]-esti[i,3]]))
        
    error_pos = np.array(error_pos)
    error_pos = np.multiply(error_pos, error_pos)
    error_pos = np.sum(error_pos, axis=1)
    # print(np.sum(np.sqrt(error_pos))/error_pos.shape[0]) 
    
    
    f_new_esti.write("# timestamp tx ty tz qx qy qz qw")
    f_new_esti.write('\r\n')
    for data in new_esti_state:
        f_new_esti.write(' '.join(data))
        f_new_esti.write('\r\n')
    f_new_esti.close()
    
    
    eulers_gt = np.array(eulers_gt)
    eulers = np.array(eulers)
    fig1, ax1 = plt.subplots(3, 3)
    
    # draw
    ax1[0][0].plot(esti[:end,0], esti[:end,1]-esti[0,1]+gt[0,1], 'b-', label = 'esti')
    ax1[0][1].plot(esti[:end,0], esti[:end,2]-esti[0,2]+gt[0,2], 'b-', label = 'esti')
    ax1[0][2].plot(esti[:end,0], esti[:end,3]-esti[0,3]+gt[0,3], 'b-', label = 'esti')
    
    ax1[1][0].plot(esti[:end,0], eulers[:end,0]-eulers[0,0]+eulers_gt[0,0], 'b-', label = 'esti')
    ax1[1][1].plot(esti[:end,0], eulers[:end,1]-eulers[0,1]+eulers_gt[0,1], 'b-', label = 'esti')
    ax1[1][2].plot(esti[:end,0], eulers[:end,2]-eulers[0,2]+eulers_gt[0,2], 'b-', label = 'esti')
    
    ax1[2][0].plot(esti[:end,0], esti[:end,8], 'b-', label = 'esti')
    ax1[2][1].plot(esti[:end,0], esti[:end,9], 'b-', label = 'esti')
    ax1[2][2].plot(esti[:end,0], esti[:end,10], 'b-', label = 'esti')
    
    
    
    ax1[0][0].plot(gt[:end,0], gt[:end,1], 'r-', label = 'gt')
    ax1[0][1].plot(gt[:end,0], gt[:end,2], 'r-', label = 'gt')
    ax1[0][2].plot(gt[:end,0], gt[:end,3], 'r-', label = 'gt')
    
    ax1[1][0].plot(gt[:end,0], eulers_gt[:end,0], 'r-', label = 'gt')
    ax1[1][1].plot(gt[:end,0], eulers_gt[:end,1], 'r-', label = 'gt')
    ax1[1][2].plot(gt[:end,0], eulers_gt[:end,2], 'r-', label = 'gt')
    
    ax1[2][0].plot(gt[:end,0], gt[:end,8], 'r-', label = 'gt')
    ax1[2][1].plot(gt[:end,0], gt[:end,9], 'r-', label = 'gt')
    ax1[2][2].plot(gt[:end,0], gt[:end,10], 'r-', label = 'gt')
    
    ax1[2][0].plot(esti[:end,0], esti[:end,11], 'g-', label = 'opti')
    ax1[2][1].plot(esti[:end,0], esti[:end,12], 'g-', label = 'opti')
    ax1[2][2].plot(esti[:end,0], esti[:end,13], 'g-', label = 'opti')
    
    ax1[0, 0].set_title("position x(m)")
    ax1[0, 1].set_title("position y(m)")
    ax1[0, 2].set_title("position z(m)")
    
    ax1[1, 0].set_title("roll(rad)")
    ax1[1, 1].set_title("pitch(rad)")
    ax1[1, 2].set_title("yaw(rad)")
    
    ax1[2, 0].set_title("velocity x(m/s)")
    ax1[2, 1].set_title("velocity y(m/s)")
    ax1[2, 2].set_title("velocity z(m/s)")
    
    lines, labels = fig1.axes[-1].get_legend_handles_labels()
    fig1.legend(lines, labels, loc = 'upper right') # 图例的位置
    # plt.legend()
    fig1.tight_layout()
    plt.savefig(save_position_path, dpi=300)

    fig2, ax2 = plt.subplots(4, 3)
    ax2[0][0].plot(bias[start:end,0], bias[start:end,1], 'b-')
    ax2[0][1].plot(bias[start:end,0], bias[start:end,2], 'b-')
    ax2[0][2].plot(bias[start:end,0], bias[start:end,3], 'b-')
    
    ax2[1][0].plot(bias[start:end,0], bias[start:end,4], 'b-')
    ax2[1][1].plot(bias[start:end,0], bias[start:end,5], 'b-')
    ax2[1][2].plot(bias[start:end,0], bias[start:end,6], 'b-')  
    
    # ax2[1][0].plot(bias[start:end,0], np.ones((end - start)) * -0.002153, 'r-')
    # ax2[1][1].plot(bias[start:end,0], np.ones((end - start)) * 0.020744, 'r-')
    # ax2[1][2].plot(bias[start:end,0], np.ones((end - start)) * 0.075806, 'r-')  
 
    # ax2[1][0].plot(bias[start:end,0], np.ones((end - start)) * -0.0546303, 'r-')
    # ax2[1][1].plot(bias[start:end,0], np.ones((end - start)) * 0.0208792, 'r-')
    # ax2[1][2].plot(bias[start:end,0], np.ones((end - start)) * 0.094797, 'r-') 
    
    # -0.002341, 0.021815, 0.07660
    ax2[1][0].plot(bias[start:end,0], np.ones((end - start)) * -0.002341, 'r-', label = 'gt')
    ax2[1][1].plot(bias[start:end,0], np.ones((end - start)) * 0.021815, 'r-', label = 'gt')
    ax2[1][2].plot(bias[start:end,0], np.ones((end - start)) * 0.07660, 'r-', label = 'gt')
    
    ax2[0][0].plot(bias[start:end,0], bias[start:end,7], 'g-', label = 'opti_bias')
    ax2[0][1].plot(bias[start:end,0], bias[start:end,8], 'g-', label = 'opti_bias')
    ax2[0][2].plot(bias[start:end,0], bias[start:end,9], 'g-', label = 'opti_bias')
    
    ax2[1][0].plot(bias[start:end,0], bias[start:end,10], 'g-', label = 'opti_bias')
    ax2[1][1].plot(bias[start:end,0], bias[start:end,11], 'g-', label = 'opti_bias')
    ax2[1][2].plot(bias[start:end,0], bias[start:end,12], 'g-', label = 'opti_bias')  
    
    ax2[2][0].plot(bias[start:end,0], bias[start:end,13], 'g-')
    
    if(use_msckf):
        ax2[0][0].plot(msckf_bias[start:end,0], msckf_bias[start:end,1], 'y-', label = 'msckf')
        ax2[0][1].plot(msckf_bias[start:end,0], msckf_bias[start:end,2], 'y-', label = 'msckf')
        ax2[0][2].plot(msckf_bias[start:end,0], msckf_bias[start:end,3], 'y-', label = 'msckf')
        
        ax2[1][0].plot(msckf_bias[start:end,0], msckf_bias[start:end,4], 'y-', label = 'msckf')
        ax2[1][1].plot(msckf_bias[start:end,0], msckf_bias[start:end,5], 'y-', label = 'msckf')
        ax2[1][2].plot(msckf_bias[start:end,0], msckf_bias[start:end,6], 'y-', label = 'msckf')
    
    ax2[0, 0].set_title("acc_x(m^s-2)")
    ax2[0, 1].set_title("acc_y(m^s-2)")
    ax2[0, 2].set_title("acc_z(m^s-2)")
    
    ax2[1, 0].set_title("gyro_x(rad/s)")
    ax2[1, 1].set_title("gyro_y(rad/s)")
    ax2[1, 2].set_title("gyro_z(rad/s)")
    
    ax2[2, 0].set_title("imu used to integrate")
    ax2[2, 1].set_title("imu buffer size")
    ax2[2, 2].set_title("features used to optimize")
    
    ax2[3, 0].set_title("bias_a(m^s-2)")
    ax2[3, 1].set_title("bias_w(rad/s)")
    
    lines, labels = fig2.axes[-1].get_legend_handles_labels()
    fig2.legend(lines, labels, loc = 'upper right') # 图例的位置
    fig2.tight_layout()
    plt.savefig(save_bias_path, dpi=300)
    
    plt.show()
  
    

if __name__ == '__main__':

    main()
    
