#ifndef _ROS_DATA_H_
#define _ROS_DATA_H_
#include <mutex>
#include <time.h>
#include <vector>
#include <queue>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Imu.h>
#include "cv_bridge/cv_bridge.h"
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#define IMUDATASIZE 10
using namespace std;
class get_ros_data
{
private:
    ros::NodeHandle nd;
    ros::Subscriber sub_color_image;
    ros::Subscriber sub_aligeddepth_image;
    cv::Mat depth_img;
    ofstream z_output_file;
    ofstream l1_output_file;
    ofstream l2_output_file;
    ofstream theta_output_file;
    Eigen::AngleAxisd r_v;
    // ofstream normal_file;
    // ofstream z_file;
    uint64_t seq;
    int index_image;

    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image> slamSyncPolicy;
    // message_filters::Subscriber<sensor_msgs::Image>* color_sub_ ;             // topic1 输入
    // message_filters::Subscriber<sensor_msgs::Image>* depth_sub_;   // topic2 输入
    // message_filters::Synchronizer<slamSyncPolicy>* sync_;

public:
    get_ros_data(ros::NodeHandle & node_);
    ~get_ros_data();
    void depthCallback(const sensor_msgs::Image::ConstPtr& msg);
    // void colorCallback(const sensor_msgs::Image::ConstPtr& msg);
    // void combineCallback(const sensor_msgs::ImageConstPtr& p_colorImage, const sensor_msgs::ImageConstPtr& p_depthImage);
};

#endif