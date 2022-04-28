#ifndef _UBUNTU_PERCEPTION_H_
#define _UBUNTU_PERCEPTION_H_
#include <time.h>
#include <vector>
#include <sys/time.h>
#include <queue>
#include <opencv2/opencv.hpp> 
#include <glog/logging.h>
// #define REALTIME
using namespace std;
class ubuntu_perception
{
public:  
    // 对于非实时直线行走和对准行走，使用一个矩形来存储台阶
    // vector<vector<Eigen::Vector3d>> planes;
    priority_queue<rectPlane> planes;
    // 对于实时行走，存储成多边形形式
    vector<taijie> mapget;
    vector<footstep> steps_result;
    vector<std::pair<Eigen::Vector3d, vector<Eigen::Vector3d>>> result;
    // 坐标转换矩阵随时更改
    double BaseVisionZ;
    double BaseVisionX;
    double BaseVisionY;
    double BaseVisionPitchDeg ;

    double base_x;
    double base_y;
    double base_height;
    double base_vx;
    double base_vy;
    double base_vz;
    // roll Pitch yaw
    double base_posture[3];
    // base_posture[0] = _deg2rad(-0.52); base_posture[1] = _deg2rad(-1.54); base_posture[2] = 0.0;

    foot_cor front_foot_cor;
    foot_cor end_foot_cor;
    foot_cor next_foot_cor;
    // 步行方式分为
    // A 直线非实时行走
    // B 对准行走
    // C 直线实时行走 
    char walk_case;
    cv::Mat depth_img_needed;
    std::vector<sensor_msgs::Imu> imuDatasNeeded;
public:
    time_self win_time;
    uint32_t seq_needed;
    ros::Time camera_time_needed;
    
    std::queue<sensor_msgs::Imu> imuDatasTmp;
    // time_self camera_time_needed;
    bool getData();
    Eigen::Matrix<double,4,4> refreshTF_W2B();
    void clearMatrixData();
    ubuntu_perception();
    ~ubuntu_perception() = default;
    Eigen::Matrix<double,4,4> refreshTF_W2B(Eigen::Matrix<double,4,4> & Matrix_C2B);
    bool plane_detection(std::ofstream& fs);
    bool stepplaning(std::ofstream& fs);
};

#endif