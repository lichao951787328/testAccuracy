#include "ubuntu_perception_noros.h"
#include "lshaped_fitting.h"
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h> 
#include "cv_bridge/cv_bridge.h"
#include <pcl/surface/concave_hull.h>
#include <Eigen/Eigenvalues> 
#include <omp.h>
#include "plane_detection.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h> 
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <unistd.h>
#include <filesystem>
#define DEBUG
string debug_dir = "/home/humanoid/catkin_bhr_ws/src/mutithread_for_walk/debugdata/";
// string debug_dir = "/home/humanoid/catkin_bhr_ws/src/mutiThread_perception/debugdata/";
ubuntu_perception::ubuntu_perception()
{
  BaseVisionZ = +1.1258e+00-+7.1365e-01+0.18938;
  BaseVisionX = 0.15995;
  BaseVisionY = 0.0;
  BaseVisionPitchDeg = 27.5;
  base_vx =0.0; base_vy =0.0; base_vz = 0.0;
  base_x = 0.0; base_y = 0.0; base_height = 0.0;
  base_posture[0] = 0.0; base_posture[1] = 0.0; base_posture[2] = 0.0; 
}

// 输入cloud
// 输出属于统一平面的点云
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

bool ubuntu_perception::getData()
{
  unique_lock<mutex> g_(m_cameradata, std::defer_lock);
  g_.lock();
  imuDatasTmp = imuDatas;
  depth_img_needed = depth_img;
  camera_time_needed = camera_time;
  seq_needed = seq;
  g_.unlock();
  return true;
}
// 一旦触发检测，得到相机数据，就需要做这个
Eigen::Matrix<double,4,4> ubuntu_perception::refreshTF_W2B(Eigen::Matrix<double,4,4> & Matrix_C2B)
{
  // 更新camera_timestd::queue<sensor_msgs::Imu> imuDatasTmp;
  time_self camera_time(camera_time_needed);
  while (!imuDatasTmp.empty())
  {
    sensor_msgs::Imu tmpIMU_head = imuDatasTmp.front();
    time_self tmpIMU_time(tmpIMU_head.header.stamp);
    imuDatasTmp.pop();
    if (tmpIMU_time - win_time > 0) //检测头,来自wintime的时间
    {
      imuDatasNeeded.emplace_back(tmpIMU_head);
      while (!imuDatasTmp.empty())
      {
        sensor_msgs::Imu tmpIMU_tail = imuDatasTmp.front();
        time_self tmpIMU_tailtime(tmpIMU_tail.header.stamp);
        imuDatasTmp.pop();
        if (tmpIMU_tailtime - camera_time < 0)// 检测尾，来自camera的时间
        {
          imuDatasNeeded.emplace_back(tmpIMU_tail);
        }
      }
    }
  }
  cout<<"win time is "<<win_time.secs<<" "<<win_time.usecs<<endl;
  cout<<"the imu need is "<<imuDatasNeeded.size()<<endl;
  for (auto & iter_imudata : imuDatasNeeded)
  {
    cout<<"imu time: "<<iter_imudata.header.stamp<<endl;
  }
  cout<<"camera time is "<<camera_time.secs<<" "<<camera_time.usecs<<endl;
  
  // 将imu数据转到世界坐标系
  base_state state(win_time, imuDatasNeeded, Matrix_C2B.block<3,3>(0,0));
  state.initial(base_x, base_y, base_height, base_vx, base_vy, base_vz, base_posture[0], base_posture[1], base_posture[2]);
  state.update(seq_needed);
  cout<<"in function refreshTF_W2B, x: "<<state.getPosition()(0)<<" y: "<<state.getPosition()(1)<<" z: "<<state.getPosition()(2)<<endl;
  return state.getMatrix4d();
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
  std::vector<cv::Point2f> vertices = lshaped.getRectVertex();
  vector<Eigen::Vector3d> edgePoints;
  for (auto & iter: vertices)
  {
    Eigen::Vector3d point(iter.x, iter.y, 0.0);
    edgePoints.emplace_back(T1*point);
  }
  return edgePoints;
}

bool ubuntu_perception::plane_detection(std::ofstream& fs)
{
  result.clear();
  steps_result.clear();
  mapget.clear(); 
  while (!planes.empty())
  {
    planes.pop();
  }
  fs<<"BaseVisionPitchDeg: "<<BaseVisionPitchDeg<<endl;
  Eigen::Matrix<double,4,4> Base_T_Vision, Vision_T_Tar, World_T_Base, World_T_Tar;
  
  
  // 由于实时需要，在此加一个
  // 输入，相机时刻，发出指令时刻，imudata
  
  // World_T_Base = getT(-0.021746375, 0.0, 0.60419, _deg2rad(-0.52),_deg2rad(-1.54), 0.0);
  Base_T_Vision = getT(BaseVisionX, BaseVisionY, BaseVisionZ, 0, 0, 0);
  Eigen::Matrix3d Base_R_VisionTemp;
  Base_R_VisionTemp << 0,-1,0, -1,0,0, 0,0,-1;
  // Base_R_VisionTemp << 0,-1,0, 1,0,0, 0,0,-1;
  // cout<<"Base_R_VisionTemp = "<<endl<<Base_R_VisionTemp<<endl;
  Base_T_Vision.block<3,3>(0,0) = Base_R_VisionTemp*(Eigen::AngleAxisd(_deg2rad(BaseVisionPitchDeg),Eigen::Vector3d::UnitX())).matrix();
#ifdef REALTIME
  World_T_Base = refreshTF_W2B(Base_T_Vision);
#else
  // World_T_Base = getT(-0.021746375, 0.0, 0.60419, _deg2rad(-0.52),_deg2rad(-1.54), 0.0);// 仅供自己测试使用
  World_T_Base = getT(base_x, base_y, base_height, base_posture[0], base_posture[1], base_posture[2]);
#endif
  Vision_T_Tar.setIdentity();
  World_T_Tar = World_T_Base*Base_T_Vision*Vision_T_Tar;
  LOG(INFO)<<"has get matrix"<<endl;
  PlaneDetection plane_detection_bl;
  plane_detection_bl.cloud.w = depth_img_needed.cols;
  plane_detection_bl.cloud.h = depth_img_needed.rows;
  plane_detection_bl.cloud.vertices.reserve(depth_img_needed.cols*depth_img_needed.rows);
  int vertex_idx = 0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
  // cout<<"World_T_Tar : "<<endl<<World_T_Tar<<endl;
  // Eigen::Matrix3d rot = World_T_Tar.block<3,3>(0,0);
  // cout<<"rot: "<<endl<<rot<<endl;
  // Eigen::Vector3d euler_angle = rot.eulerAngles(2,1,0);
  // cout<<"euler_angle: "<<euler_angle.transpose()<<endl;
  // 在做平面检测时，需要排除front_foot.x之前的点
  int rows = depth_img_needed.rows;
  int cols = depth_img_needed.cols;
  LOG(INFO)<<"rows : "<<rows<<" cols : "<<cols<<endl;
	#pragma omp parallel for reduction(+:plane_detection_bl, vertex_idx, source)
  for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
      // LOG(INFO)<<"the filter"<<endl;
			double z = (double)(depth_img_needed.at<unsigned short>(i, j)) / kScaleFactor;
			if (_isnan(z))
			{
				plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
				continue;
			}
			double x = ((double)j - kCx) * z / kFx;
			double y = ((double)i - kCy) * z / kFy;
      Eigen::Vector3d point(x, y, z);
      Eigen::Vector4d point_;
      point_.head(3) = point;
      point_(3) = 1;
      Eigen::Vector3d point_trans = (World_T_Tar*point_).head(3);
      point_trans(2) -= 0.032;
      point_trans(2) += 0.0117713;
      
      // 过滤地面 确定地面坐标，是世界坐标系即地面0，可能为负？？？
      // 对于这两个过滤阈值，point_trans(2) < 0.034是为了滤除地面，而point_trans(2) > 0.1是为了滤除高于台阶的物体，例如机器人本体的膝盖和手臂
      // 可根据
      if (point_trans(2) > 0.2)
      {
        plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(0.0, 0.0, 0.0f/0.0f);
        continue;
      }
      
      if (walk_case == 'A')// 直线行走
      {
        // cout<<"no real time straight walk filter..."<<endl;
        // LOG(INFO)<<"no real time straight walk filter..."<<endl;
        if (point_trans(0) < 0.3 || point_trans(0) > 1.5)
        {
          plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(0.0, 0.0, 0.0f/0.0f);
          continue;
        }

        if(point_trans(1) > 0.1 || point_trans(1) < - 0.1)
        {
          plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(0.0, 0.0, 0.0f/0.0f);
          continue;
        }

        pcl::PointXYZ p(point_trans(0), point_trans(1), point_trans(2));
        source->emplace_back(p);
        plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(point_trans(0), point_trans(1), point_trans(2));
      }
      else if (walk_case == 'B' || walk_case == 'C')// 对准行走
      {
        // cout<<"no real time aim walk filter..."<<endl;
        if (point_trans(2) < 0.04)//滤去地面, 地面阈值设定
        {
          plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(0.0, 0.0, 0.0f/0.0f);
          continue;
        }
        pcl::PointXYZ p(point_trans(0), point_trans(1), point_trans(2));
        source->emplace_back(p);
        plane_detection_bl.cloud.vertices[vertex_idx++] = VertexType(point_trans(0), point_trans(1), point_trans(2));
      }
      else
      {

      }
		}
	}
  LOG(INFO)<<"filter depth over"<<endl;
  LOG(INFO)<<"filted size is "<<source->size()<<endl;
  if (source->size() > 0)
  {
    string save_pcd_path_point = debug_dir + std::to_string(seq_needed) + "_" +".pcd";
    pcl::io::savePCDFileASCII(save_pcd_path_point, *source);
  }
  else
  {
    cout<<"the point cloud is zero."<<endl;
    return false;  
  }
    
  // finish = clock();
  // cout<<"data time is "<<(double)(finish - start)/CLOCKS_PER_SEC <<endl;
#ifdef DEBUG
  // std::cout<<"cloud v size is "<<plane_detection_bl.cloud.vertices.size()<<std::endl;
  // pcl::visualization::CloudViewer viewer ("Cluster viewer");
  // viewer.showCloud(source);
  // system("read -p 'Press Enter to continue...' var");
  // std::cout<<"start to run plane_detection"<<std::endl;
#endif
  plane_detection_bl.runPlaneDetection();
#ifdef DEBUG
  fs<<"plaeeeeeeeee "<<plane_detection_bl.plane_num_<<std::endl;
  cout<<"plaeeeeeeeee "<<plane_detection_bl.plane_num_<<std::endl;
  // cout<<plane_detection_bl.seg_img_.cols< " "<<plane_detection_bl.seg_img_.rows<<" nowwwwwwwwwwwwww"<<endl;
  if (plane_detection_bl.plane_num_ > 0)
  {
    cv::imwrite(debug_dir + std::to_string(seq_needed) + ".png", plane_detection_bl.seg_img_);
  }
  else
  {
    LOG(INFO)<<"no plane has detected..."<<endl;
  }
  
  // uint32_t sum = 0.0;
  // for(size_t i = 0; i < plane_detection_bl.plane_vertices_.size(); i++)
  // {
  //     sum += plane_detection_bl.plane_vertices_.at(i).size();
  // }
  // cout<<"sum point cloud is "<<sum<<endl;
  // std::cout<<"plaeeeeeeeee "<<plane_detection_bl.plane_vertices_.size()<<std::endl;
  // cout<<"plane size is "<<plane_detection_bl.plane_vertices_.size()<<endl;
#endif
  result.clear();
  result.reserve(plane_detection_bl.plane_vertices_.size());
  
  // coverTaijie.clear();
  // coverTaijie.reserve(plane_detection_bl.plane_vertices_.size());
  // 遍历每个平面
  for (size_t i = 0; i < plane_detection_bl.plane_vertices_.size(); i++)
  {
    vector<int>& tmpIndexs = plane_detection_bl.plane_vertices_.at(i);
    // vector<Eigen::Vector3d>& tmpPoints = result.at(i);
    int sum_num = tmpIndexs.size();
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmpCloud(new pcl::PointCloud<pcl::PointXYZ>);
    tmpCloud->reserve(sum_num);
#ifdef DEBUG
    fs<<"plane "<<i+1<<" has "<<sum_num<<"points."<<std::endl;
#endif 
// 平面内的点太少，舍弃
    if(sum_num < 1000)
    {
      cout<<"the number in this plan is less than 1000, discard the plan."<<endl;
      continue;
    }
      
    double sumx = 0, sumy = 0, sumz = 0;
    vector<int>::const_iterator iter_int;
    for(iter_int = tmpIndexs.begin(); iter_int != tmpIndexs.end(); iter_int++)
    {
      double x, y, z;
      plane_detection_bl.cloud.get(*iter_int, x, y, z);
      sumx += x; sumy += y; sumz += z;
      pcl::PointXYZ point(x,y,z);
      tmpCloud->emplace_back(point);
    }
#ifdef DEBUG1
    string save_pcd_path = std::to_string(seq) + "_" + std::to_string(i) +".pcd";
    pcl::io::savePCDFileASCII(save_pcd_path, *tmpCloud);
#endif
    // pcl::visualization::CloudViewer viewer ("Cluster viewer");
    // viewer.showCloud(tmpCloud);
    // system("read -p 'Press Enter to continue...' var");
    string save_pcd_path = debug_dir + std::to_string(seq_needed) + "_" + std::to_string(i) +".pcd";
    pcl::io::savePCDFileASCII(save_pcd_path, *tmpCloud);
    // 加一个体素滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(tmpCloud);//给滤波对象设置需过滤的点云
    sor.setLeafSize(0.005f, 0.005f, 0.005f);//设置滤波时创建的体素大小为1cm*1cm*1cm的立方体
    sor.filter(*cloud_filtered);//执行滤波处理，存储输出为cloud_filtered

    /*
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);

    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);

        // 把平面内点提取到一个新的点云中
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::ExtractIndices<pcl::PointXYZ> ex ;
    ex.setInputCloud(cloud_filtered);
    ex.setIndices(inliers);
    ex.filter(*cloud_plane);
    */
    // 精度。小于7mm，1m以内的测量距离
    // 精度z方向 1000 + 14.5 - 44 + 4.5 = 975
    // 975 - 30 = 945
    // 角度在相机与测量平行（用手机水平仪测量有1度的差别）的情况下，平面角度测量角度精度小于1.7度
    // 获取了多个平面的外轮廓
    pcl::PointXYZ tmpCenter(sumx/sum_num, sumy/sum_num, sumz/sum_num);
    cout<<"center "<<i+1<<" z = "<<tmpCenter.z<<endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_points(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Vector3d normal;
    getPlanePoints(plane_points, normal, cloud_filtered, tmpCenter);
    string save_plane_pc_path = debug_dir + std::to_string(seq_needed) + "_" + std::to_string(i) +"planepc.pcd";
    pcl::io::savePCDFileASCII(save_plane_pc_path, *plane_points);
    LOG(INFO)<<"save plane points."<<endl;
    cout<<"normal is "<<normal(0)<<" "<<normal(1)<<" "<<normal(2)<<endl;
    double theta = acos(abs(normal(2)))*57.3;
    fs<<"theta : "<<theta<<endl;
    if(theta > 30)
    {
      cout<<"the plane angle is too big. disgard..."<<endl;
      continue;
    }
    // std::cout<<"get plane points"<<std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConcaveHull<pcl::PointXYZ> chull;
    chull.setInputCloud(plane_points);
    chull.setAlpha(10);
    chull.reconstruct(*cloud_hull);
    vector<Eigen::Vector3d> poly;
    poly.reserve(cloud_hull->size());

    // 此处开始分流，需不需要对实时和非实时的情况进行处理
#ifdef REALTIME
    for (auto & iter_hullpoint : *cloud_hull)
    {
      poly.emplace_back(Eigen::Vector3d(iter_hullpoint.x, iter_hullpoint.y, iter_hullpoint.z));
    }
    result.emplace_back(make_pair(normal, poly));
#else
    Eigen::Vector3d center_eigen(tmpCenter.x, tmpCenter.y, tmpCenter.z);
    vector<Eigen::Vector3d> rectPoint = fitRect(*cloud_hull, normal, center_eigen);

    // 检查矩形滤波是否正确
    string save_rect_path = debug_dir + std::to_string(seq_needed) + "_" + std::to_string(i) +"rect.pcd";
    for (size_t k = 0; k < 4; k++)
    {
      pcl::PointXYZ p(rectPoint.at(k)(0), rectPoint.at(k)(1), rectPoint.at(k)(2));
      cloud_hull->emplace_back(p);
    }
    LOG(INFO)<<"save rect cloud"<<endl;
    // pcl::visualization::CloudViewer viewer ("Cluster viewer");
    // viewer.showCloud(cloud_hull);
    // system("read -p 'Press Enter to continue...' var");
    pcl::io::savePCDFileASCII(save_rect_path, *cloud_hull);
    LOG(INFO)<<"edge size "<<rectPoint.size()<<endl;
    rectPlane tmprect(rectPoint);
    LOG(INFO)<<"tmprect center: "<<tmprect.center.transpose()<<endl;
    planes.push(tmprect);
#endif
  }
  // priority_queue<rectPlane> planes_copy = planes;
  // while (!planes_copy.empty())
  // {
  //   rectPlane p = planes.top();
  //   planes.pop();
  //   LOG(INFO)<<"PLANE CENTER:"<<endl;
  //   LOG(INFO)<<p.center.transpose()<<endl;
  //   LOG(INFO)<<"plane corners: "<<endl;
  //   for (size_t i = 0; i < 4; i++)
  //   {
  //     LOG(INFO)<<p.corners[i].transpose()<<endl;
  //   }
  //   LOG(INFO)<<endl;
  // }
  
  
  // A 直线非实时行走
  // B 对准行走
  // C 直线实时行走 
  if (walk_case == 'A')
  {
    cout<<"case A : no real time straight walk..."<<endl;
    foot_data footparam;
    footparam.fit_length = 0.2;
    footparam.foot_end_length = 0.09;
    footparam.foot_front_length = 0.15;
    footparam.max_length = 0.4;
    footparam.min_length = 0.03;
    footparam.threld = 0.04;
    vector<taijie> Taijies;
    vector<taijie> input_map;
    if (!planes.empty())
    {
      // 收集检测到的台阶
      while (!planes.empty())
      {
        rectPlane p = planes.top();
        planes.pop();
        double minx = 10;
        double maxx = - 10;
        double sumz = 0.0;
        for (size_t i = 0; i < 4; i++)
        {
          if (minx > p.corners[i](0))
          {
            minx = p.corners[i](0);
          }
          if (maxx < p.corners[i](0))
          {
            maxx = p.corners[i](0);
          }
          sumz += p.corners[i](2);
        }
        double z = sumz/4;
        cout<<"the z is "<<z<<endl;
        if (z > 0.030 && z < 0.055)
        {
          cout<<"the z is change to 0.045 "<<endl;
          z = 0.045;
        }
        else if (abs(z) < 0.02)
        {
          cout<<"the z is change to 0.0 "<<endl;
          z = 0.0;
        }
        if (maxx - minx < footparam.foot_end_length + footparam.foot_front_length)
        {
          LOG(INFO)<<"the length of taijie is less than foot length, discard it."<<endl;
          continue;
        }
        taijie tmptaijie;
        tmptaijie.start = minx;
        if(!Taijies.empty())
        {
          tmptaijie.start = Taijies.back().end;
        }
        tmptaijie.end = maxx;
        // tmptaijie.z = 0.045;
        tmptaijie.height = z;
        Taijies.push_back(tmptaijie);
      }
      
      // 转存  先不考虑复杂的情况
      if (Taijies.empty())
      {
        LOG(INFO)<<"no taijie"<<endl;
        return false;
      }
      else
      {
        input_map = Taijies;
      }
    }
    else
    {
      LOG(INFO)<<"no plane detected, check..."<<endl;
      return false;
    }
    if (input_map.empty())
    {
      LOG(INFO)<<"the input map is empty, return false."<<endl;
      return false;
    }
    cout<<"the input taijie message is "<<endl;
    for (auto & iter_taijie : input_map)
    {
      LOG(INFO)<<iter_taijie.start<<" "<<iter_taijie.end<<" "<<iter_taijie.height<<endl;
    }
    
    if (input_map.back().height != 0.0)
    {
      taijie butaijie;
      double start = 0.0;
      if(!input_map.empty())
      {
        LOG(INFO)<<"SOME TAIJIE IN MAP"<<endl;
        start = input_map.back().end;
      }
      butaijie.start = start;
      butaijie.end = start + 0.5;
      butaijie.height = 0.0;
      input_map.emplace_back(butaijie);
    }
    else if(input_map.back().end - input_map.back().start < 0.5)
    {
      LOG(WARNING)<<"the last plane is too short, extend it to 0.5m..."<<endl;
      input_map.back().end = input_map.back().start + 0.5;
    }
    LOG(INFO)<<"THE MAP IN STRAIGHT WALK STEP PLANING"<<endl;
    for (auto & iter_taijie : input_map)
    {
      LOG(INFO)<<iter_taijie.start<<" "<<iter_taijie.end<<" "<<iter_taijie.height<<endl;
    }
    
    straight_walk straightWalker(input_map, footparam);
    straightWalker.go();
    steps_result = straightWalker.getSteps();
    for (auto & iter : steps_result)
    {
      if (iter.is_left)
      {
        ROS_INFO("%s, %f, %f\n", "left : ", iter.x, iter.y);
      }
      else
        ROS_INFO("%s, %f, %f\n", "right : ", iter.x, iter.y);
    }
    return steps_result.size() > 0;
  }
  else if(walk_case == 'B')
  {
    LOG(INFO)<<"case B : no real time aim walk..."<<endl;
    if (!planes.empty())
    {
      LOG(INFO)<<"plane corners is "<<endl;
      for (size_t i = 0; i < 4; i++)
      {
        LOG(INFO)<<"coner "<<i+1<<" : "<<planes.top().corners[i].transpose()<<endl;
      }
      // 对检测到的台阶进行排序，找到最近的
      Eigen::Vector3d direct = planes.top().direct;
      Eigen::Vector3d point = planes.top().center;
      double width_2d = planes.top().width_2d;
      LOG(INFO)<<"aim direct "<<direct.transpose()<<endl;
      LOG(INFO)<<"plane center is "<<point.transpose()<<endl;
      LOG(INFO)<<"plane width is "<<width_2d<<endl;
      // 直线方程为 a(x-x0)+b(y-y0)=0
      double goal_x = point(0) + point(1)*direct(1)/direct(0);
      Eigen::Vector2d goal_2d(goal_x, 0.0);
      double dis = (goal_2d - point.head(2)).norm() - width_2d/2.0;
      if (dis > 0.4)//沿着机器人方向有交点，需要保证机器人下蹲之后，能看到地面
      {
        LOG(INFO)<<"just need to walk straight"<<endl;
        double theta = acos(Eigen::Vector3d::UnitX().dot(direct));
        aim_stepplanning aim_walk(goal_2d, direct.head(2), 0.25);
        aim_walk.go();
        steps_result = aim_walk.getResult();
        return steps_result.size()>0;
      }
      else
      {
        LOG(INFO)<<"need to turn direct at stand still"<<endl;
        Eigen::Vector2d goal_2d_vitual = point.head(2) - direct.head(2).normalized() * (width_2d/2.0 + 0.4);
        Eigen::Vector2d direct_vitual = goal_2d_vitual.normalized();// 机器人第一阶段转向
        // 要将机器人旋转到这个方向, theta永远为正
        double theta = acos(Eigen::Vector2d::UnitX().dot(direct_vitual));
        CHECK_GE(theta, 0.0)<<"theta is > 0"<<endl;
        LOG(ERROR)<<"THETA : "<<theta<<endl;
        double angle = 5/57.3;
        int num_angle = (int)(theta/angle);
        if (theta - angle*num_angle > (2/57.3))
        {
          num_angle ++;
        }
        double turn_angle = theta/num_angle;
        vector<footstep> turn_steps;
        bool turn_flag;
        if (direct_vitual(1) < 0)//右转
        {
          turn_flag = false;//false 右转
        }
        else
        {
          turn_flag = true;
        }
        LOG(INFO)<<"the turn need "<<num_angle*2<<" steps, and turn "<<turn_flag<<endl;
        // 固定旋转的计算方式存在问题，很大问题
        double x = 0.0, y = 0.0;
        // 左转theta角大于0
        if (turn_flag)//左转先左脚，theta角大于0
        {
          for (size_t i = 0; i < num_angle; i++)
          {
            double theta_i = (i+1)*turn_angle;
            Eigen::Vector3d left_foot_cor(0.0, 0.08, 0.0);
            Eigen::Vector3d right_foot_cor(0.0, -0.08, 0.0);
            Eigen::AngleAxisd rotate_vector(theta_i, Eigen::Vector3d::UnitZ());
            Eigen::Vector3d left_foot_cor_ro = rotate_vector.toRotationMatrix() * left_foot_cor;
            Eigen::Vector3d right_foot_cor_ro = rotate_vector.toRotationMatrix() * right_foot_cor;
            footstep tmpstep1;
            tmpstep1.is_left = true;
            tmpstep1.x = left_foot_cor_ro(0);
            tmpstep1.y = left_foot_cor_ro(1);
            tmpstep1.z = left_foot_cor_ro(2);
            tmpstep1.theta = theta_i;
            turn_steps.emplace_back(tmpstep1);
            footstep tmpstep2;
            tmpstep2.is_left = false;
            tmpstep1.x = right_foot_cor_ro(0);
            tmpstep1.y = right_foot_cor_ro(1);
            tmpstep1.z = right_foot_cor_ro(2);
            tmpstep2.theta = theta_i;
            turn_steps.emplace_back(tmpstep2);
          }
        }
        else// 右转theta角小于0
        {
          for (size_t i = 0; i < num_angle; i++)
          {
            double theta_i = - (i+1)*turn_angle;
            Eigen::Vector3d left_foot_cor(0.0, 0.08, 0.0);
            Eigen::Vector3d right_foot_cor(0.0, -0.08, 0.0);
            Eigen::AngleAxisd rotate_vector(theta_i, Eigen::Vector3d::UnitZ());
            Eigen::Vector3d left_foot_cor_ro = rotate_vector.toRotationMatrix() * left_foot_cor;
            Eigen::Vector3d right_foot_cor_ro = rotate_vector.toRotationMatrix() * right_foot_cor;
            footstep tmpstep1;
            tmpstep1.is_left = false;
            tmpstep1.x = right_foot_cor_ro(0);
            tmpstep1.y = right_foot_cor_ro(1);
            tmpstep1.z = right_foot_cor_ro(2);
            tmpstep1.theta = theta_i;
            turn_steps.emplace_back(tmpstep1);
            footstep tmpstep2;
            tmpstep2.is_left = true;
            tmpstep2.x = left_foot_cor_ro(0);
            tmpstep2.y = left_foot_cor_ro(1);
            tmpstep2.z = left_foot_cor_ro(2);
            tmpstep2.theta = theta_i;
            turn_steps.emplace_back(tmpstep2);
          }
        }
        LOG(INFO)<<"turn steps is: "<<endl;
        for (auto & iter_step : turn_steps)
        {
          LOG(INFO)<<"step: "<<iter_step.is_left<<" "<<iter_step.x<<" "<<iter_step.y<<" "<<iter_step.z<<" "<<iter_step.theta<<endl;
        }
        aim_stepplanning aim_walk(goal_2d_vitual, direct.head(2), 0.25);
        aim_walk.go();
        vector<footstep> walk_steps = aim_walk.getResult();
        steps_result = turn_steps;
        LOG(INFO)<<"result steps is: "<<endl;
        for (auto & iter_step : walk_steps)
        {
          LOG(INFO)<<"step: "<<iter_step.is_left<<" "<<iter_step.x<<" "<<iter_step.y<<" "<<iter_step.z<<" "<<iter_step.theta<<endl;
        }
        for (auto & iter_step : walk_steps)
        {
          // world to local 
          Eigen::AngleAxisd rotate_vector_W2L(theta, Eigen::Vector3d::UnitZ());
          Eigen::Vector3d tmp_step(iter_step.x, iter_step.y, iter_step.z);
          Eigen::Vector3d tmp_step_rot = rotate_vector_W2L.toRotationMatrix().inverse() * tmp_step;
          Eigen::AngleAxisd rotate_vector_L2F(iter_step.theta, Eigen::Vector3d::UnitZ());
          Eigen::Matrix3d r_W2F = rotate_vector_W2L.toRotationMatrix() * rotate_vector_L2F.toRotationMatrix();
          // yaw pitch roll
          Eigen::Vector3d eulerAngle = r_W2F.eulerAngles(2,1,0);
          footstep tmpstep;
          tmpstep.is_left = iter_step.is_left;
          tmpstep.x       = tmp_step_rot(0);
          tmpstep.y       = tmp_step_rot(1);
          tmpstep.z       = tmp_step_rot(2);
          tmpstep.theta   = eulerAngle(0);
          steps_result.emplace_back(tmpstep);
        }
        return steps_result.size() > 0;
      }
    }
    else
    {
      LOG(INFO)<<"no plane detected in walk case B, check..."<<endl;
      return false;
    }
  }
  else if(walk_case == 'C')
  {
    // 20220415对准靠近代码
    LOG(INFO)<<"case C : no real time aim walk and close to TAIJIE..."<<endl;
    if (!planes.empty())
    {
      LOG(INFO)<<"plane corners is "<<endl;
      for (size_t i = 0; i < 4; i++)
      {
        LOG(INFO)<<"coner "<<i+1<<" : "<<planes.top().corners[i].transpose()<<endl;
      }
      // 对检测到的台阶进行排序，找到最近的
      Eigen::Vector3d direct = planes.top().direct;
      Eigen::Vector3d point = planes.top().center;
      double width_2d = planes.top().width_2d;
      Eigen::Vector2d direct_2d = direct.head(2).normalized();
      Eigen::Vector2d line_point = point.head(2) - direct_2d*width_2d/2.0;
      // double x = line_point(1)*direct_2d(1)/direct_2d(0) + line_point(0);
      double dis = abs(line_point.dot(direct_2d));
      double goal_dis = dis - 0.16;//前脚长15cm + 1cm阈值
      LOG(INFO)<<"aim direct "<<direct_2d.transpose()<<endl;
      LOG(INFO)<<"goal distance : "<<goal_dis<<endl;
      double theta = acos(Eigen::Vector2d::UnitX().dot(direct_2d));
      CHECK_GE(theta, 0.0)<<"theta is > 0"<<endl;
      LOG(ERROR)<<"THETA : "<<theta<<endl;
      struct line_step
      {
        double x, y, theta;
      };
      if (direct_2d(1) < 0)//右转
      {
        LOG(INFO)<<"TURN RIGHT ..."<<endl;
        int num_foot_len = (int)(goal_dis/0.25 + 0.8);
        int num_foot_angle = (int)(abs(theta)/(6/57.3) + 0.8);
        int num_foot = max(num_foot_len, num_foot_angle);
        double length_step = goal_dis / num_foot;
        double theta_step = theta / num_foot;
        LOG(INFO)<<"step length "<<length_step<<endl;
        LOG(INFO)<<"step angle "<<theta_step<<endl;
        vector<line_step> line_steps;
        line_steps.reserve(num_foot);
        // footstep tmpstep;
        // tmpstep.x = 0.0;
        // tmpstep.y = 0.0;
        // tmpstep.z = 0.0;
        // tmpstep.theta = 0.0;
        // steps_result.emplace_back(tmpstep);
        for (size_t i = 0; i < num_foot; i++)
        {
          Eigen::Vector2d line_cor = length_step *(i+1) *direct_2d;
          double tmptheta = (i+1) * theta_step;
          line_step tmp_line_step;
          tmp_line_step.x = line_cor(0);
          tmp_line_step.y = line_cor(1);
          tmp_line_step.theta = tmptheta;
          line_steps.emplace_back(tmp_line_step);
          // footstep tmpstep;
          // tmpstep.x = line_cor(0);
          // tmpstep.y = line_cor(1);
          // tmpstep.z = 0.0;
          // tmpstep.theta = -tmptheta;
          // steps_result.emplace_back(tmpstep);
        }
        LOG(INFO)<<"line steps :"<<endl;
        for (auto & iter_line_step : line_steps)
        {
          LOG(INFO)<<iter_line_step.x<<" "<<iter_line_step.y<<" "<<iter_line_step.theta<<endl;
        }
        // return steps_result.size() > 0;
        for (auto & iter_line_step : line_steps)
        {
          Eigen::Vector3d t(iter_line_step.x, iter_line_step.y, 0.0);
          Eigen::Vector3d left_foot_cor(0.0, 0.08, 0.0);
          Eigen::Vector3d right_foot_cor(0.0, -0.08, 0.0);
          Eigen::AngleAxisd rotate_vector( - iter_line_step.theta, Eigen::Vector3d::UnitZ());
          Eigen::Vector3d left_foot_cor_ro = rotate_vector.toRotationMatrix() * left_foot_cor + t;
          Eigen::Vector3d right_foot_cor_ro = rotate_vector.toRotationMatrix() * right_foot_cor + t;
          footstep tmpstep1;
          tmpstep1.is_left = false;
          tmpstep1.x = right_foot_cor_ro(0);
          tmpstep1.y = right_foot_cor_ro(1);
          tmpstep1.z = right_foot_cor_ro(2);
          tmpstep1.theta = - iter_line_step.theta;
          steps_result.emplace_back(tmpstep1);
          footstep tmpstep2;
          tmpstep2.is_left = true;
          tmpstep2.x = left_foot_cor_ro(0);
          tmpstep2.y = left_foot_cor_ro(1);
          tmpstep2.z = left_foot_cor_ro(2);
          tmpstep2.theta = - iter_line_step.theta;
          steps_result.emplace_back(tmpstep2);
        }
        LOG(INFO)<<"foot step:"<<endl;
        for (auto & iter_footstep : steps_result)
        {
          LOG(INFO)<<iter_footstep.is_left<<" "<<iter_footstep.x<<" "<<iter_footstep.y<<" "<<iter_footstep.z<<" "<<iter_footstep.theta<<endl;
        }
        
      }
      else//左转
      {
        LOG(INFO)<<"TURN LEFT ..."<<endl;
        int num_foot_len = (int)(goal_dis/0.25 + 0.8);
        int num_foot_angle = (int)(abs(theta)/(6/57.3) + 0.8);
        int num_foot = max(num_foot_len, num_foot_angle);
        double length_step = goal_dis / num_foot;
        double theta_step = theta / num_foot;
        LOG(INFO)<<"step length "<<length_step<<endl;
        LOG(INFO)<<"step angle "<<theta_step<<endl;
        vector<line_step> line_steps;
        line_steps.reserve(num_foot);
        // footstep tmpstep;
        // tmpstep.x = 0.0;
        // tmpstep.y = 0.0;
        // tmpstep.z = 0.0;
        // tmpstep.theta = 0.0;
        // steps_result.emplace_back(tmpstep);
        for (size_t i = 0; i < num_foot; i++)
        {
          Eigen::Vector2d line_cor = length_step *(i+1) *direct_2d;
          double tmptheta = (i+1) * theta_step;
          line_step tmp_line_step;
          tmp_line_step.x = line_cor(0);
          tmp_line_step.y = line_cor(1);
          tmp_line_step.theta = tmptheta;
          line_steps.emplace_back(tmp_line_step);
          // footstep tmpstep;
          // tmpstep.x = line_cor(0);
          // tmpstep.y = line_cor(1);
          // tmpstep.z = 0.0;
          // tmpstep.theta = tmptheta;
          // steps_result.emplace_back(tmpstep);
        }
        for (auto & iter_line_step : line_steps)
        {
          LOG(INFO)<<iter_line_step.x<<" "<<iter_line_step.y<<" "<<iter_line_step.theta<<endl;
        }
        // return steps_result.size() > 0;
        for (auto & iter_line_step : line_steps)
        {
          Eigen::Vector3d t(iter_line_step.x, iter_line_step.y, 0.0);
          Eigen::Vector3d left_foot_cor(0.0, 0.08, 0.0);
          Eigen::Vector3d right_foot_cor(0.0, -0.08, 0.0);
          Eigen::AngleAxisd rotate_vector(iter_line_step.theta, Eigen::Vector3d::UnitZ());
          Eigen::Vector3d left_foot_cor_ro = rotate_vector.toRotationMatrix() * left_foot_cor + t;
          Eigen::Vector3d right_foot_cor_ro = rotate_vector.toRotationMatrix() * right_foot_cor + t;
          footstep tmpstep1;
          tmpstep1.is_left = true;
          tmpstep1.x = left_foot_cor_ro(0);
          tmpstep1.y = left_foot_cor_ro(1);
          tmpstep1.z = left_foot_cor_ro(2);
          tmpstep1.theta = iter_line_step.theta;
          steps_result.emplace_back(tmpstep1);
          footstep tmpstep2;
          tmpstep2.is_left = false;
          tmpstep2.x = right_foot_cor_ro(0);
          tmpstep2.y = right_foot_cor_ro(1);
          tmpstep2.z = right_foot_cor_ro(2);
          tmpstep2.theta = iter_line_step.theta;
          steps_result.emplace_back(tmpstep2);
        }
        LOG(INFO)<<"foot step:"<<endl;
        for (auto & iter_footstep : steps_result)
        {
          LOG(INFO)<<iter_footstep.is_left<<" "<<iter_footstep.x<<" "<<iter_footstep.y<<" "<<iter_footstep.z<<" "<<iter_footstep.theta<<endl;
        }
      }
      return steps_result.size() > 0;
    }
    else
    {
      LOG(INFO)<<"no plane detected in walk case C, check..."<<endl;
      return false;
    }
    return true;
  }
}

void ubuntu_perception::clearMatrixData()
{
  base_x = 0.0; base_y = 0.0; base_height = 0.0;
  base_posture[0] = 0.0; base_posture[1] = 0.0; base_posture[2] =0.0;
}

// 下面这个是实时规划的函数
/*
bool ubuntu_perception::stepplaning(std::ofstream& fs)
{
  mapget.clear();
  for (size_t i = 0; i < map.size(); i++)
  {
    taijie tmp;
    tmp.start = map.at(i).start;
    tmp.end = map.at(i).end;
    tmp.height = map.at(i).height;
    mapget.emplace_back(tmp);
  }
  steplong step_information;
  // 前长0.14 后长0.1
  // 本身残留的步态 记录下一步，
  // 初始时刻，
  fs<<"********************** important information in stepplanning******************"<<endl;
  fs<<"map in stepplanning : "<<endl;
  for (size_t i = 0; i < mapget.size(); i++)
  {
    fs<<"satrt: "<<mapget.at(i).start<<" end: "<<mapget.at(i).end<<" height: "<<mapget.at(i).height<<endl;
  }
  fs<<"******************** important information in stepplanning*******************"<<endl<<endl;
  stepplaning_obj step_gen(step_information, mapget, 0.14, 0.1, 0.015);
  step_gen.set_current_front_foot(front_foot_cor.is_left, front_foot_cor.cor[0], front_foot_cor.cor[2]);
  step_gen.set_current_end_foot(end_foot_cor.is_left, end_foot_cor.cor[0], end_foot_cor.cor[2]);
  step_gen.set_next_foot(next_foot_cor.is_left, next_foot_cor.cor[0], next_foot_cor.cor[2]);
  cout<<"enter stepplanning go function."<<endl;
  steps_result = step_gen.go(fs);
  if(!steps_result.empty())
  {
    // check the step is good
    bool check_ok = true;
    for (size_t i = 0; i < steps_result.size() - 1; i++)
    {
      bool flag_dis = steps_result.at(i + 1).x - steps_result.at(i).x > 0.32;
      bool flag_height = abs(steps_result.at(i + 1).z - steps_result.at(i).z) > 0.6;
      if (flag_dis || flag_height)
      {
        check_ok = false;
        break;
      }
      
    }
    
    if (!check_ok)
    {
      fs<<"the planning steps is wrong, please check."<<endl;
      for (size_t i = 0; i < steps_result.size(); i++)
      {
        fs<<"step "<<i+1<<" "<<steps_result.at(i).is_left<<" "<<steps_result.at(i).x<<" "<<steps_result.at(i).z<<endl;
      }
      return false;
    }
    cout<<"function stepplanning go is right."<<endl;
    return true;
  }
  else
  {
    cout<<"function stepplanning go is wrong."<<endl;
    return false;
  }
}
*/

