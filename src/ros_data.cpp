#include "ros_data.h"
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
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "lshaped_fitting.h"
#include <queue>
#include <omp.h>
#include <assert.h>
#define resolution_angle_i 0.00179
#define resolution_angle_j 0.0021
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
  BaseVisionZ = +1.1258e+00-+7.1365e-01+0.18938;
  // 对于新的工装
  BaseVisionZ -= 0.06435;
  BaseVisionX = 0.15995;
  // 对于新的工装
  BaseVisionX -= 0.00298;
  BaseVisionY = 0.0;
  BaseVisionPitchDeg = 27.5;

  
  index_image = 1;
  r_v = Eigen::AngleAxisd(_Degree2Rad(33), Eigen::Vector3d::UnitX());
  
  // cv::namedWindow(OPENCV_WINDOW);
  LOG(INFO)<<"..."<<endl;
  color_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(nd, "/camera/color/image_raw", 1);
  // depth_sub_  = new message_filters::Subscriber<sensor_msgs::Image>(nd, "/camera/aligned_depth_to_color/image_raw", 1);
  depth_sub_  = new message_filters::Subscriber<sensor_msgs::Image>(nd, "/camera/aligned_depth_to_color/image_raw", 1);

  sync_ = new  message_filters::Synchronizer<slamSyncPolicy>(slamSyncPolicy(10), *color_sub_, *depth_sub_);
  sync_->registerCallback(boost::bind(&get_ros_data::combineCallback,this, _1, _2));
  LOG(INFO)<<"initial ros finish..."<<endl;

}

get_ros_data::~get_ros_data()
{
  // cv::destroyWindow(OPENCV_WINDOW);
}

void getPlanePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr planepoints, Eigen::Vector3d& normal, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ& center)
{
  planepoints->reserve(cloud->size());
  // std::cout<<"1"<<std::endl;
  Eigen::Matrix3d M = Eigen::Matrix3d::Zero(3,3);
  // std::cout<<"1"<<std::endl;
  // pcl::PointCloud<pcl::PointXYZ>::iterator index = cloud->begin();
  pcl::PointCloud<pcl::PointXYZ>::iterator iter_point;
  // std::cout<<"1"<<std::endl;
  for (iter_point = cloud->begin(); iter_point != cloud->end(); iter_point++)
  {   
    // std::cout<<"2"<<std::endl;
    Eigen::Vector3d ve((*iter_point).x - center.x, (*iter_point).y - center.y, (*iter_point).z - center.z);
    M += ve*ve.transpose();
  }
  // std::cout<<"get M matrix"<<std::endl;
  Eigen::EigenSolver<Eigen::Matrix3d> es(M);
  Eigen::Matrix3d::Index b;
  auto minEigenValue = es.eigenvalues().real().minCoeff(&b);
  double eigenValuesSum = es.eigenvalues().real().sum();
  normal = es.eigenvectors().real().col(b);
  // std::cout<<"get normal"<<std::endl;
  Eigen::Vector3d center_(center.x, center.y, center.z);
  double d = -(normal.dot(center_));
  
  for (iter_point = cloud->begin(); iter_point != cloud->end(); iter_point++)
  {
    Eigen::Vector3d point((*iter_point).x, (*iter_point).y, (*iter_point).z);
    double dis = normal.dot(point) + d;
    Eigen::Vector3d pointShape = point - dis*normal;
    pcl::PointXYZ p(pointShape(0), pointShape(1), pointShape(2));
    planepoints->emplace_back(p);
  }
  // std::cout<<"get plane points"<<std::endl;
}

vector<Eigen::Vector3d> fitRect(pcl::PointCloud<pcl::PointXYZ>& cloud_hull, Eigen::Vector3d & normal, Eigen::Vector3d & center_eigen)
{
  Eigen::Vector3d z_axid = normal;
  Eigen::Vector3d x_point;
  for (auto & iter:(cloud_hull))
  {
    Eigen::Vector3d tmppoint(iter.x, iter.y, iter.z);
    if ((tmppoint - center_eigen).norm() > 0.2)
    {
        x_point = tmppoint;
        break;
    }
  }
  cout<<"the cor point is "<<x_point(0)<<" "<<x_point(1)<<" "<<x_point(2)<<endl;
  Eigen::Vector3d x_axid = (x_point - center_eigen).normalized();
  Eigen::Vector3d y_axid = (normal.cross(x_axid)).normalized();

  cout<<"x : "<<x_axid.transpose()<<endl;
  cout<<"y : "<<y_axid.transpose()<<endl;
  cout<<"z : "<<z_axid.transpose()<<endl;
  // 从定义的平面坐标系到世界坐标系
  Eigen::Matrix3d rotation2W;

  rotation2W<<x_axid.dot(Eigen::Vector3d::UnitX()), y_axid.dot(Eigen::Vector3d::UnitX()), 
              z_axid.dot(Eigen::Vector3d::UnitX()), x_axid.dot(Eigen::Vector3d::UnitY()),
              y_axid.dot(Eigen::Vector3d::UnitY()), z_axid.dot(Eigen::Vector3d::UnitY()),
              x_axid.dot(Eigen::Vector3d::UnitZ()), y_axid.dot(Eigen::Vector3d::UnitZ()),
              z_axid.dot(Eigen::Vector3d::UnitZ());
  Eigen::Isometry3d T1=Eigen::Isometry3d::Identity();
  T1.rotate (rotation2W);
  T1.pretranslate (center_eigen);
  std::vector<cv::Point2f> hull;
  for (auto & iter:(cloud_hull))
  {
      Eigen::Vector3d new_p = T1.inverse()*Eigen::Vector3d(iter.x, iter.y, iter.z);
      hull.emplace_back(cv::Point2f(new_p(0), new_p(1)));
  }
  LShapedFIT lshaped;
  cv::RotatedRect rr = lshaped.FitBox(&hull);
  LOG(INFO)<<"MIN COV IS: "<<lshaped.min_cov<<endl; 
  vector<Eigen::Vector3d> edgePoints;
  edgePoints.clear();
  if (lshaped.min_cov > 0.01)
  {
    LOG(INFO)<<"THE MIN COV IS large, the shape may be not rect..."<<endl;
    return edgePoints;
  }
  
  std::vector<cv::Point2f> vertices = lshaped.getRectVertex();
  for (auto & iter: vertices)
  {
    Eigen::Vector3d point(iter.x, iter.y, 0.0);
    edgePoints.emplace_back(T1*point);
  }
  return edgePoints;
}

bool inRange(double x, double t1, double t2)
{
  return x >= t1 && x <= t2;
}

auto getT(const double &px, const double &py, const double &pz, const double &rx, const double &ry, const double &rz)
{
  using namespace Eigen;
  Matrix4d res;
  res.setIdentity();
  res.block<3,3>(0,0) = (AngleAxisd(rz, Vector3d::UnitZ())*AngleAxisd(ry, Vector3d::UnitY())*AngleAxisd(rx, Vector3d::UnitX())).matrix();
  res(0,3) = px;
  res(1,3) = py;
  res(2,3) = pz;
  return res;
}

auto _deg2rad(double degree)
{
  double rad = degree/57.3;
  return rad;
}

void get_ros_data::combineCallback(const sensor_msgs::ImageConstPtr& p_colorImage, const sensor_msgs::ImageConstPtr& p_depthImage)
{
  LOG(INFO)<<p_colorImage->encoding<<endl;
  // cv::Mat perspective_image = cv_bridge::toCvShare(p_colorImage, sensor_msgs::image_encodings::RGB8)->image;
  // cv::imwrite(debug_dir + std::to_string(index_image) + "color.png", perspective_image);
  // index_image++;
  // return;
  cv_bridge::CvImageConstPtr cv_ptr;
  cv::Mat hsv_image;
  
  cv::Mat gray_image;
  cv::Mat Gaussian_image;
  cv::Mat threld_image;
  cv::Mat pz_image;
	try
	{
		cv::Mat perspective_image = cv_bridge::toCvShare(p_colorImage, sensor_msgs::image_encodings::RGB8)->image;
    LOG(INFO)<<cv_bridge::toCvShare(p_colorImage, sensor_msgs::image_encodings::RGB8)->encoding<<endl;
		if(!perspective_image.empty())
		{
      LOG(INFO)<<"get mat"<<endl;
      LOG(INFO)<<perspective_image.rows<<" "<<perspective_image.cols<<endl;
      LOG(INFO)<<perspective_image.at<cv::Vec3b>(0, 0)<<endl;
      // cv::imshow("color image", perspective_image);
      LOG(INFO)<<"SHOW COLOR IMAGE"<<endl;
      cv::cvtColor(perspective_image, gray_image, cv::COLOR_RGB2GRAY);
      LOG(INFO)<<"change to gray image"<<endl;
      cv::GaussianBlur(gray_image, Gaussian_image, cv::Size(7, 7), 0, 0);
      LOG(INFO)<<"change to Gaussian image"<<endl;
      // 80 是晚上的阈值，洗干净之后，颜色更黑，阈值可以降的更低
      cv::threshold(Gaussian_image, threld_image, 100, 255, cv::THRESH_BINARY);
      cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
      cv::dilate(threld_image, pz_image, element);
      cv::imwrite(debug_dir + std::to_string(index_image) + "threld_image.png", threld_image);
		}
    cv::waitKey(1);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s'.", p_colorImage->encoding.c_str());
	}
  // index_image++;
  // return;
  // 白色为255
  LOG(INFO)<<(int)threld_image.at<uchar>(0,0)<<endl;
  LOG(INFO)<<(int)threld_image.at<uchar>(0,1)<<endl;
  cv::imwrite(debug_dir + std::to_string(index_image) + "threld_image.png", threld_image);
  
  LOG(INFO)<<cv_bridge::toCvShare(p_depthImage, sensor_msgs::image_encodings::TYPE_16UC1)->encoding<<endl;
  depth_img = cv_bridge::toCvShare(p_depthImage, sensor_msgs::image_encodings::TYPE_16UC1)->image;
  LOG(INFO)<<"get aligned depth image"<<endl;

  size_t rows = depth_img.rows;
  size_t cols = depth_img.cols;
  LOG(INFO)<<"ROWS: "<<rows<<" COLS: "<<cols<<endl;
  PlaneDetection plane_detection_bl;
  plane_detection_bl.cloud.w = depth_img.cols;
  plane_detection_bl.cloud.h = depth_img.rows;
  plane_detection_bl.cloud.vertices.reserve(depth_img.cols*depth_img.rows);
  uint32_t vertex_idx = 0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr threld_filter_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr need_could(new pcl::PointCloud<pcl::PointXYZ>);
  // 相机视角是足够大的，需要过滤一部分，以免外部的点对其造成干扰
  size_t rows_start, rows_end, cols_start, cols_end;
  // 不同的图像长宽不同，阈值不同
  rows_start = 0;
  rows_end = rows;
  cols_start = 0;
  cols_end = cols;
  assert(rows_start <= rows);
  assert(rows_end <= rows);
  assert(cols_start <= cols);
  assert(cols_end <= cols);
  Eigen::Matrix<double,4,4> World_T_Base, Base_T_Vision, VisionL515_T_VisionD435i, VisionD435i_T_Target, World_T_Tar;
  Base_T_Vision = getT(BaseVisionX, BaseVisionY, BaseVisionZ, 0, 0, 0);
  Eigen::Matrix3d Base_R_VisionTemp;
  Base_R_VisionTemp << 0,-1,0, -1,0,0, 0,0,-1;
  Base_T_Vision.block<3,3>(0,0) = Base_R_VisionTemp*(Eigen::AngleAxisd(_deg2rad(BaseVisionPitchDeg),Eigen::Vector3d::UnitX())).matrix();
  double base_x = 0.0;
  double base_y = 0.0;
  double base_height = 0.65;
  double base_posture[3] = {0.0};
  World_T_Base = getT(base_x, base_y, base_height, base_posture[0], base_posture[1], base_posture[2]);
  VisionL515_T_VisionD435i = getT(-0.0325, 0.01865, 0.00575, 0, 0, 0);
  VisionD435i_T_Target.setIdentity();
  World_T_Tar = World_T_Base*Base_T_Vision*VisionL515_T_VisionD435i*VisionD435i_T_Target;
  LOG(INFO)<<"has get matrix"<<endl;

  #pragma omp parallel for reduction(+:plane_detection_bl, vertex_idx, raw_cloud, threld_filter_cloud)
  for (size_t i = rows_start; i < rows_end; i++)//height
  {
    for (size_t j = cols_start; j < cols_end; j++)// width
    {
      assert(vertex_idx < depth_img.cols*depth_img.rows);
      double z = (double)(depth_img.at<unsigned short>(i, j)) / kScaleFactor;
			if (_isnan(z))
			{
				plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
				continue;
			}
      double x = ((double)j - kCx) * z / kFx;
			double y = ((double)i - kCy) * z / kFy;
      plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(x, y, z);
      Eigen::Vector3d point(x,y,z);
      Eigen::Vector4d point_;
      point_.head(3) = point;
      point_(3) = 1;
      Eigen::Vector3d point_trans = (World_T_Tar*point_).head(3);
      pcl::PointXYZ pclpoint(point_trans(0), point_trans(1), point_trans(2));
      raw_cloud->emplace_back(pclpoint);
      if ((int)threld_image.at<uchar>(i,j) > 100)// 如果这个满足某个条件
      {
        plane_detection_bl.cloud.vertices[vertex_idx] = VertexType(0, 0, 0.0f/0.0f);
				continue;
      }
      threld_filter_cloud->emplace_back(pclpoint);
    }
  }
  pcl::visualization::CloudViewer viewer ("viewer");
  viewer.showCloud(raw_cloud);
  system("read -p 'Press Enter to continue...' var");

  string save_raw_cloud_path = debug_dir + std::to_string(index_image) +"rawcloud.pcd";
  pcl::io::savePCDFileASCII(save_raw_cloud_path, *raw_cloud);

  string save_threld_cloud_path = debug_dir + std::to_string(index_image) +"threldcloud.pcd";
  pcl::io::savePCDFileASCII(save_threld_cloud_path, *threld_filter_cloud);
  index_image++;
  return;
  // 可以再加一个飞点滤除
  /*
  pcl::PointCloud<pcl::PointXYZ>::Ptr filter_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  #pragma omp parallel for reduction(+:plane_detection_bl, filter_cloud)
  for (size_t i = rows_start; i < rows_end; i++)//height
  {
    for (size_t j = cols_start; j < cols_end; j++)// width
    {
      // 根据D435I的FOV，
      // (i,j)点的角度
      // double theta_i = i * resolution_angle_i - 0.75922;
      // double theta_j = j * resolution_angle_j - 0.506145;
      // Eigen::Vector3d direct_spray(sin(theta_i)*sin(theta_j), sin(theta_i)*cos(theta_j), cos(theta_i));// 这一步后续可以使用查表进行
      bool check_flag_i = (i == rows_start) || (i == rows_end-1);
      bool check_flag_j = (j == cols_start) || (j == cols_end-1);
      // 可以设置最外层一圈均不在考虑范围内
      if (!check_flag_i && !check_flag_j)
      {
        vector<Eigen::Vector3d> points;
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        points.reserve(5);
        double x, y, z;
        plane_detection_bl.cloud.get(i, j, x, y, z);

        // plane_detection_bl.cloud.get(j*rows + i, x, y, z);
        if (_isnan(z))
        {
          continue;
        }
        Eigen::Vector3d center_point(x, y, z);
        Eigen::Vector3d direct_spray = center_point.normalized();
        center += center_point;
        points.emplace_back(center_point);
        plane_detection_bl.cloud.get(i-1, j, x, y, z);
        if (_isnan(z))
        {
          continue;
        }
        Eigen::Vector3d point1(x, y, z);
        center += point1;
        points.emplace_back(point1);
        plane_detection_bl.cloud.get(i+1, j, x, y, z);
        if (_isnan(z))
        {
          continue;
        }
        Eigen::Vector3d point2(x, y, z);
        center += point2;
        points.emplace_back(point2);
        plane_detection_bl.cloud.get(i, j -1, x, y, z);
        if (_isnan(z))
        {
          continue;
        }
        Eigen::Vector3d point3(x, y, z);
        center += point3;
        points.emplace_back(point3);
        plane_detection_bl.cloud.get(i, j + 1, x, y, z);
        if (_isnan(z))
        {
          continue;
        }
        Eigen::Vector3d point4(x, y, z);
        center += point4;
        points.emplace_back(point4);
        center /= 5;
        Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
        for (auto & iter_point : points)
        {
          Eigen::Vector3d c_p = iter_point - center;
          M += c_p * c_p.transpose();
        }
        Eigen::EigenSolver<Eigen::Matrix3d> es(M);
        Eigen::Matrix3d::Index b;
        auto minEigenValue = es.eigenvalues().real().minCoeff(&b);
        double eigenValuesSum = es.eigenvalues().real().sum();
        Eigen::Vector3d normal = es.eigenvectors().real().col(b);
        if (abs(normal.dot(direct_spray)) < 0)// 这个阈值需要调节
        {
          // 最好是等到所有飞点全部确定之后再一次性全部更新，这样不会影响别的点进行法向量估计
          
          plane_detection_bl.cloud.vertices[i*rows + j] = VertexType(0, 0, 0.0f/0.0f);
          continue;
        }
      }
      // 可以在此处加坐标转换，再进行过滤
      // 如果不加，就是检测点云内所有平面
      double x,y,z;
      plane_detection_bl.cloud.get(i, j, x, y, z);
      pcl::PointXYZ point(x,y,z);
      filter_cloud->emplace_back(point);
    }
  }

  string save_filter_cloud_path = debug_dir + std::to_string(index_image) +"filtercloud.pcd";
  pcl::io::savePCDFileASCII(save_filter_cloud_path, *filter_cloud);
*/
  
  plane_detection_bl.runPlaneDetection();
  LOG(INFO)<<"plane detection over..."<<endl;
  LOG(INFO)<<"contains "<<plane_detection_bl.plane_vertices_.size()<<"planes..."<<endl;
  cv::imwrite(debug_dir + std::to_string(index_image) + "plane.png", plane_detection_bl.seg_img_);
  // return;
  for (size_t i = 0; i < plane_detection_bl.plane_vertices_.size(); i++)
  {
    ofstream points_file_seq(debug_dir + std::to_string(index_image) +"_" + std::to_string(i) + "points.txt");
    ofstream normal_file_seq(debug_dir + std::to_string(index_image) +"_" + std::to_string(i) + "normal.txt");
    ofstream z_file_seq(debug_dir + std::to_string(index_image) +"_" + std::to_string(i) + "z.data");
    vector<int>& tmpIndexs = plane_detection_bl.plane_vertices_.at(i);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    double sumx = 0, sumy = 0, sumz = 0;
    vector<int>::const_iterator iter_int;
    for(iter_int = tmpIndexs.begin(); iter_int != tmpIndexs.end(); iter_int++)
    {
      double x, y, z;
      plane_detection_bl.cloud.get(*iter_int, x, y, z);
      z_file_seq<<z<<endl;
      sumx += x; sumy += y; sumz += z;
      pcl::PointXYZ point(x,y,z);
      cloud_filtered->emplace_back(point);
    }
    string save_pcd_path = debug_dir + std::to_string(index_image) +"_" + std::to_string(i) +".pcd";
    pcl::io::savePCDFileASCII(save_pcd_path, *cloud_filtered);

    int sum_num = cloud_filtered->size();
    z_output_file<<sumz/sum_num<<endl;
    pcl::PointXYZ tmpCenter(sumx/sum_num, sumy/sum_num, sumz/sum_num);

    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_points(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Vector3d normal;
    getPlanePoints(plane_points, normal, cloud_filtered, tmpCenter);
    string save_plane_pc_path = debug_dir + std::to_string(index_image) + "_" + std::to_string(i) + "planepc.pcd";
    pcl::io::savePCDFileASCII(save_plane_pc_path, *plane_points);
    LOG(INFO)<<"save plane points."<<endl;
    cout<<"normal is "<<normal(0)<<" "<<normal(1)<<" "<<normal(2)<<endl;
    double theta = acos(abs(normal(2)))*57.3;
    theta_output_file<<theta<<endl;
    if(theta > 30)
    {
      cout<<"the plane angle is too big. disgard..."<<endl;
      return ;
    }
    // std::cout<<"get plane points"<<std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConcaveHull<pcl::PointXYZ> chull;
    chull.setInputCloud(plane_points);
    chull.setAlpha(10);
    chull.reconstruct(*cloud_hull);
    Eigen::Vector3d center_eigen(tmpCenter.x, tmpCenter.y, tmpCenter.z);
    vector<Eigen::Vector3d> rectPoint = fitRect(*cloud_hull, normal, center_eigen);
    if (theta < 30 && rectPoint.size() == 4)
    {
      normal_file_seq<<normal.transpose()<<endl;
      for (auto & iter_point : rectPoint)
      {
        points_file_seq<<iter_point.transpose()<<endl;
      }
    }
    else
    {
      LOG(INFO)<<"THE THETA IS "<<theta<<endl;
      LOG(INFO)<<"the edge points size is "<<rectPoint.size()<<endl;
    }
    LOG(INFO)<<"index_image: "<<index_image<<endl;
    // double l1 = (rectPoint.at(1) - rectPoint.at(0)).norm();
    // double l2 = (rectPoint.at(2) - rectPoint.at(1)).norm();
    // l1_output_file<<min(l1, l2)<<endl;
    // l2_output_file<<max(l1, l2)<<endl;
  }
  LOG(INFO)<<"RECORD OVER"<<endl;
  index_image++;
}

