#include "ros_data_testdepthdata.h"
#include <iostream>
#include <glog/logging.h>
using namespace std;

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]); 
  FLAGS_colorlogtostderr = true;
  FLAGS_log_dir = "./log"; 
  FLAGS_alsologtostderr = true;
  LOG(INFO)<<"initial glog finish"<<endl;
  ros::init(argc, argv,"plane_detection");
  ros::NodeHandle nh;
  get_ros_data get_data_node(nh);
  ros::spin();
  return 0;
}