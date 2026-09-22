#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <Eigen/Dense>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

struct cuCam {
	int width;
	int height;
	float fx;
	float fy;
	float cx;
	float cy;
	float k1;
	float k2;
	float p1;
	float p2;
	float k3;
};

struct cuPose {
	float R[9];   // world->cam
	float t[3];
};

// Adjoint matrix per node, 6x6 row-major
// | R [t]x_R |
// | 0    R   |
struct cuAdj {
	float data[36];
};
inline Eigen::Matrix3f skew_symmetric(const Eigen::Vector3f& v)
{
	Eigen::Matrix3f m;
	m << 0, -v.z(), v.y(),
		v.z(), 0, -v.x(),
		-v.y(), v.x(), 0;
	return m;
}

struct cuRelPose {
	float R[9];
	float R_inv[9];
	float t[3];
	float t_inv[3];
};

struct cuEdge {
	int i, j;
	const float* depth_i;
	const float3* normal_i;
	const float* depth_j;
	const float* I_i;
	const float* I_i_gx;
	const float* I_i_gy;
	const float* I_j;
	const uint8_t* mask_j; // (optional)
	cuRelPose T_ij;
	cuAdj d_AdInv;
};

// === test ===
struct SinglePairData {
	cuCam cam;
	const float* depth1;
	const float3* norm1;
	const float* depth2;
	const double2* raymap;
	cuRelPose pose;          // 当前线性化位姿（T_21）
	float* Minv_diag;     // device 上的预条件器对角（长度 6）
};

struct DenseBAData {
	cuCam cam;
	const double2* raymap;
	int numFrames;
	// 预条件器对角（按 6*numFrames 存）
	float* d_Minv_diag;
	int numEdges;
	cuEdge* d_edges;
};

inline void ensure_dir(const string& p)
{
	if (!cv::utils::fs::exists(p))
		if (!cv::utils::fs::createDirectories(p))
			std::cerr << "Failed to create dir: " << p << "\n";
}