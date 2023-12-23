#ifndef InitializeOpti_H
#define InitializeOpti_H

#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <msckf_vio/imu_state.h>

using namespace std;
using namespace Eigen;

namespace msckf_vio{
class InitializeOpti {
public:
    InitializeOpti(vector<IMUState> window): window(window){}
                                                                           
    template<typename T>
    bool operator()(const T *const q_imu_opti, T *residuals) const {
        Eigen::Quaternion<T> q(q_imu_opti[3], q_imu_opti[0], q_imu_opti[1], q_imu_opti[2]);
        q.normalize();
        for(int i = 0; i < window.size(); i++){
            IMUState imu_state = window[i];
            cout << "vel: " << imu_state.velocity.transpose() << 
                " opti: " << imu_state.opti_speed.transpose() << endl;
            Eigen::Matrix<T, 3, 1> imu_velocity = imu_state.velocity.cast<T>();
            Eigen::Matrix<T, 3, 1> opti_velocity = imu_state.opti_speed.cast<T>();
            Eigen::Matrix<T, 3, 1> error = opti_velocity - q.toRotationMatrix() * imu_velocity;
            residuals[i*3] = error[0];
            residuals[i*3+1] = error[1];
            residuals[i*3+2] = error[2];
        }     
        return true;
    }


private:
  vector<IMUState> window;
};


}
#endif // PreIntegrateReprojectError.h