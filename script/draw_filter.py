import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from tf.transformations import euler_from_quaternion

def main():
    
    foldpath = "/home/ldd/msckf_ws/src/msckf_vio/dataset/real/data_1_16_line_2/"
    gt_path = foldpath + "groundtruth_velocity.txt"
    opti_path = foldpath + "filter_velocity.txt"
    no_filter_path = foldpath + "no_filter_velocity.txt"
    
    gt_vel = np.loadtxt(gt_path, delimiter=' ', skiprows=1)
    opti_vel = np.loadtxt(opti_path, delimiter=' ', skiprows=1)
    no_filter = np.loadtxt(no_filter_path, delimiter=' ', skiprows=1)
    
    fig1, ax1 = plt.subplots(3, 1)
    gt_vel = np.array(gt_vel)
    opti_vel = np.array(opti_vel)
    
    start_id = 0
    while(np.abs(gt_vel[start_id,1]) > 1):
        start_id += 1
        
    print(gt_vel.shape[0], opti_vel.shape[0])
    
    ax1[0].plot(gt_vel[start_id:,0], gt_vel[start_id:,1], 'r-', label = 'gt')
    ax1[1].plot(gt_vel[start_id:,0], gt_vel[start_id:,2], 'r-', label = 'gt')
    ax1[2].plot(gt_vel[start_id:,0], gt_vel[start_id:,3], 'r-', label = 'gt')
    
    ax1[0].plot(opti_vel[:,0], opti_vel[:,1], 'b-', label = 'opti')
    ax1[1].plot(opti_vel[:,0], opti_vel[:,2], 'b-', label = 'opti')
    ax1[2].plot(opti_vel[:,0], opti_vel[:,3], 'b-', label = 'opti')
    
    # ax1[0].plot(no_filter[:,0], no_filter[:,1], 'g-', label = 'no_filter')    
    # ax1[1].plot(no_filter[:,0], no_filter[:,2], 'g-', label = 'no_filter')
    # ax1[2].plot(no_filter[:,0], no_filter[:,3], 'g-', label = 'no_filter')
    
    error_vel = []
    gt_num = 0
    
    while(np.abs(gt_vel[gt_num,1]) > 1):
        gt_num += 1
    is_finish = False
    for i in range(opti_vel.shape[0]):
        
        if(gt_num > gt_vel.shape[0]):
            break
        
        while(gt_vel[gt_num,0] < opti_vel[i, 0]):
            if(gt_num >= gt_vel.shape[0]-1):
                is_finish = True
                break
                
            gt_num += 1
        # print(gt_num)
        
        if(is_finish):
            break
                
        error_vel.append(np.array([gt_vel[gt_num,1]-opti_vel[i,1], gt_vel[gt_num,2]-opti_vel[i,2], gt_vel[gt_num,3]-opti_vel[i,3]]))
    
    error_vel = np.array(error_vel)
    error_vel = np.multiply(error_vel, error_vel)
    error_vel = np.sum(error_vel, axis=1)
    print("filter: ", np.sum(np.sqrt(error_vel))/error_vel.shape[0]) 
    
    
    error_vel = []
    gt_num = 0
    
    while(np.abs(gt_vel[gt_num,1]) > 1):
        gt_num += 1    
    is_finish = False
    for i in range(no_filter.shape[0]):
        
        if(gt_num > gt_vel.shape[0]):
            break
        
        while(gt_vel[gt_num,0] < no_filter[i, 0]):
            if(gt_num >= gt_vel.shape[0]-1):
                is_finish = True
                break
                
            gt_num += 1
        # print(gt_num)
        
        if(is_finish):
            break
                
        error_vel.append(np.array([gt_vel[gt_num,1]-no_filter[i,1], gt_vel[gt_num,2]-no_filter[i,2], gt_vel[gt_num,3]-no_filter[i,3]]))
    
    error_vel = np.array(error_vel)
    error_vel = np.multiply(error_vel, error_vel)
    error_vel = np.sum(error_vel, axis=1)
    print("no_filter: ", np.sum(np.sqrt(error_vel))/error_vel.shape[0])
    
    lines, labels = fig1.axes[-1].get_legend_handles_labels()
    fig1.legend(lines, labels, loc = 'upper right') # 图例的位置
    fig1.tight_layout()
    
    save_path = foldpath + "result1.png"
    plt.savefig(save_path, dpi=300)

    plt.show()
    

if __name__ == '__main__':

    main()
    
