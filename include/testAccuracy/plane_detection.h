#ifndef PLANEDETECTION_H
#define PLANEDETECTION_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
// #include <Eigen>
#include "AHCPlaneFitter.hpp"
#include "mrf.h"
#include "GCoptimization.h"
#include <unordered_map>

using namespace std;

typedef Eigen::Vector3d VertexType;
typedef Eigen::Vector2i PixelPos;

const int kNeighborRange = 5; // boundary pixels' neighbor range
// const int kScaleFactor = 5; // scale coordinate unit in mm
const int kScaleFactor = 1000; // scale coordinate unit in mm
const float kInfVal = 1000000; // an infinite large value used in MRF optimization

// Camera intrinsic parameters.
// All BundleFusion data uses the following parameters.
// const double kFx = 583;
// const double kFy = 583;
// const double kCx = 320;
// const double kCy = 240;

// L515深度图内参
// const double kFx = 460.2265625;
// const double kFy = 460.25;
// const double kCx = 325.44140625;
// const double kCy = 236.3984375;
// const int kDepthWidth = 640;
// const int kDepthHeight = 480;

// D435I深度图内参 没有将深度与彩色图对准
// const double kFx = 422.33465576171875;
// const double kFy = 422.33465576171875;
// const double kCx = 424.4653015136719;
// const double kCy = 237.94264221191406;
// const int kDepthWidth = 848;
// const int kDepthHeight = 480;

// D435I深度图内参 没有将深度与彩色图对准
// K = 【FX 0 CX
//       0 FY CY
//       0 0 1】
// const double kFx = 918.8240966796875;
// const double kFy = 917.1801147460938;
// const double kCx = 664.4627685546875;
// const double kCy = 375.1426696777344;
// const int kDepthWidth = 1280;
// const int kDepthHeight = 720;


//自己测的内参
const double kFx = 914.332756;
const double kFy = 913.209783;
const double kCx = 664.022095;
const double kCy = 372.956474;
const int kDepthWidth = 1280;
const int kDepthHeight = 720;


// aliged coloried
// const double kFx = 612.5493774414062;
// const double kFy = 611.4534301757812;
// const double kCx = 336.3084716796875;
// const double kCy = 250.09512329101562;
// const int kDepthWidth = 640;
// const int kDepthHeight = 480;

// 一般的
// const double kFx = 382.49176025390625;
// const double kFy = 382.49176025390625;
// const double kCx = 320.42138671875;
// const double kCy = 238.17295837402344;
// const int kDepthWidth = 640;
// const int kDepthHeight = 480;

#if defined(__linux__) || defined(__APPLE__)
#define _isnan(x) isnan(x)
#endif

struct ImagePointCloud
{
	vector<VertexType> vertices; // 3D vertices
	int w, h;

	inline int width() const { return w; }
	inline int height() const { return h; }
	inline bool get(const int row, const int col, double &x, double &y, double &z) const {
		const int pixIdx = row * w + col;
		z = vertices[pixIdx][2];
		// Remove points with 0 or invalid depth in case they are detected as a plane
		if (z == 0 || _isnan(z)) return false;
		x = vertices[pixIdx][0];
		y = vertices[pixIdx][1];
		return true;
	}

	inline bool get(const int index, double &x, double &y, double &z) const {
		z = vertices[index][2];
		// Remove points with 0 or invalid depth in case they are detected as a plane
		if (z == 0 || _isnan(z)) return false;
		x = vertices[index][0];
		y = vertices[index][1];
		return true;
	}
};

// Data for sum of points on a same plane
struct SumStats
{
	double sx, sy, sz; // sum of x/y/z
	double sxx, syy, szz, sxy, syz, sxz; // sum of x*x/y*y/z*z/x*y/y*z/x*z
	SumStats(){
		sx = sy = sz = sxx = syy = szz = sxy = syz = sxz = 0;
	}
};

class PlaneDetection
{
public:
	ImagePointCloud cloud;
	ahc::PlaneFitter< ImagePointCloud > plane_filter;
	vector<vector<int>> plane_vertices_; // vertex indices each plane contains
	cv::Mat seg_img_; // segmentation image
	cv::Mat color_img_; // input color image
	int plane_num_;

	/* For MRF optimization */
	cv::Mat opt_seg_img_;
	cv::Mat opt_membership_img_; // optimized membership image (plane index each pixel belongs to)
	vector<bool> pixel_boundary_flags_; // pixel is a plane boundary pixel or not
	vector<int> pixel_grayval_;
	vector<cv::Vec3b> plane_colors_;
	vector<SumStats> sum_stats_, opt_sum_stats_; // parameters of sum of points from the same plane
	vector<int> plane_pixel_nums_, opt_plane_pixel_nums_; // number of pixels each plane has
	unordered_map<int, int> pid_to_extractedpid; // plane index -> extracted plane index of plane_filter.extractedPlanes
	unordered_map<int, int> extractedpid_to_pid; // extracted plane index -> plane index

public:
	PlaneDetection();
	~PlaneDetection();

	void clearAll();
	//bool readIntrinsicParameterFile(string filename);

	bool readColorImage(string filename);

	bool readDepthImage(string filename);

	bool readDepthImageFromPCLPOINTXYZ(string filename);

	bool runPlaneDetection();

	void prepareForMRF();

	void writeOutputFiles(string output_folder, string frame_name);

	void writePlaneDataFile(string filename);

	void writePlaneLabelFile(string filename);

	/************************************************************************/
	/* For MRF optimization */
	inline MRF::CostVal dCost(int pix, int label)
	{
		return pixel_boundary_flags_[pix] ? 1 : (label == plane_filter.membershipImg.at<int>(pix / kDepthWidth, pix % kDepthWidth) ? 1 : kInfVal);
	}
	inline MRF::CostVal fnCost(int pix1, int pix2, int i, int j)
	{
		int gray1 = pixel_grayval_[pix1], gray2 = pixel_grayval_[pix2];
		return i == j ? 0 : exp(-MRF::CostVal(gray1 - gray2) * (gray1 - gray2) / 900); // 900 = sigma^2 by default
	}
	/************************************************************************/

private:
	inline int RGB2Gray(int x, int y)
	{
		return int(0.299 * color_img_.at<cv::Vec3b>(x, y)[2] +
			0.587 * color_img_.at<cv::Vec3b>(x, y)[1] +
			0.114 * color_img_.at<cv::Vec3b>(x, y)[0] +
			0.5);
	}

	void computePlaneSumStats();

};


#endif
