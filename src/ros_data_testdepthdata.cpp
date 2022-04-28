#include "ros_data_testdepthdata.h"
#include <glog/logging.h>
#include <queue>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>
#include <plane_detection.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/io/io.h>
#include <pcl/io/ascii_io.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "lshaped_fitting.h"
#include <queue>
#include <omp.h>
using namespace std;
string debug_dir = "/home/humanoid/catkin_bhr_ws/src/testAccuracy/debugdata/";
static const std::string OPENCV_WINDOW = "Image window";
auto _Degree2Rad(double d)
{
  return d/57.3;
}

get_ros_data::get_ros_data(ros::NodeHandle & node_):nd(node_)
{
  // sub_image = nd.subscribe("/camera/depth/image_rect_raw", 1, &get_ros_data::chatterCallback, this);
  // sub_aligeddepth_image = nd.subscribe("/camera/aligned_depth_to_color/image_raw", 1, &get_ros_data::depthCallback, this);
  // sub_color_image = nd.subscribe("/camera/color/image_raw", 1, &get_ros_data::colorCallback, this);
  z_output_file.open(debug_dir + "z.txt");
  l1_output_file.open(debug_dir + "l1.txt");
  l2_output_file.open(debug_dir + "l2.txt");
  theta_output_file.open(debug_dir + "theta.txt");
  index_image = 1;
  r_v = Eigen::AngleAxisd(_Degree2Rad(33), Eigen::Vector3d::UnitX());
  
  // cv::namedWindow(OPENCV_WINDOW);
  LOG(INFO)<<"..."<<endl;
  sub_aligeddepth_image = nd.subscribe("/camera/aligned_depth_to_color/image_raw", 1, &get_ros_data::depthCallback, this);
  // color_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(nd, "/camera/color/image_raw", 1);
  // // depth_sub_  = new message_filters::Subscriber<sensor_msgs::Image>(nd, "/camera/aligned_depth_to_color/image_raw", 1);
  // depth_sub_  = new message_filters::Subscriber<sensor_msgs::Image>(nd, "/camera/depth/image_rect_raw", 1);

  // sync_ = new  message_filters::Synchronizer<slamSyncPolicy>(slamSyncPolicy(10), *color_sub_, *depth_sub_);
  // sync_->registerCallback(boost::bind(&get_ros_data::combineCallback,this, _1, _2));
  LOG(INFO)<<"initial ros finish..."<<endl;

}

get_ros_data::~get_ros_data()
{
  // cv::destroyWindow(OPENCV_WINDOW);
}


void get_ros_data::depthCallback(const sensor_msgs::Image::ConstPtr& msg)
{
  LOG(INFO)<<msg->encoding<<endl;
  LOG(INFO)<<cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_16UC1)->encoding<<endl;
  depth_img = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
  size_t rows = depth_img.rows;
  size_t cols = depth_img.cols;
  LOG(INFO)<<"ROWS: "<<rows<<" COLS: "<<cols<<endl;
  PlaneDetection plane_detection_bl;
  plane_detection_bl.cloud.w = depth_img.cols;
  plane_detection_bl.cloud.h = depth_img.rows;
  plane_detection_bl.cloud.vertices.reserve(depth_img.cols*depth_img.rows);
  uint32_t vertex_idx = 0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
  size_t rows_start, rows_end, cols_start, cols_end;
  rows_start = 0;
  rows_end = rows;
  cols_start = 0;
  cols_end = cols;
  assert(rows_start <= rows);
  assert(rows_end <= rows);
  assert(cols_start <= cols);
  assert(cols_end <= cols);
  #pragma omp parallel for reduction(+:plane_detection_bl, vertex_idx, source)
  for (size_t i = rows_start; i < rows_end; i++)//height
  {
    for (size_t j = cols_start; j < cols_end; j++)// width
    {
      double z = (double)(depth_img.at<unsigned short>(i, j)) / kScaleFactor;
			if (_isnan(z))
			{
				plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
				continue;
			}
      double x = ((double)j - kCx) * z / kFx;
			double y = ((double)i - kCy) * z / kFy;
      plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(x, y, z);
      pcl::PointXYZ point(x,y,z);
      source->emplace_back(point);
    }
  }
  string save_raw_cloud_path = debug_dir + std::to_string(index_image) +"rawcloud.pcd";
  pcl::io::savePCDFileASCII(save_raw_cloud_path, *source);
  plane_detection_bl.runPlaneDetection();
  LOG(INFO)<<"plane detection over..."<<endl;
  LOG(INFO)<<"contains "<<plane_detection_bl.plane_vertices_.size()<<"planes..."<<endl;
  cv::imwrite(debug_dir + std::to_string(index_image) + "plane.png", plane_detection_bl.seg_img_);
  index_image++;
  return;
}

