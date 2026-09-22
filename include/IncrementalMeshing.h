#pragma once

// open3d_incremental_mesh.cpp
// Author: Lee
// Date: 2025-05-02
// Incremental high-precision color point cloud to mesh using Open3D C++ API

#include <open3D/Open3D.h>
#include <mutex>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace open3d;
using namespace std;

class IncrementalTSDFReconstructor {
public:
	IncrementalTSDFReconstructor(
		double voxel_length = 0.005,
		double sdf_trunc = 0.04)
		: voxel_length_(voxel_length), sdf_trunc_(sdf_trunc), tsdf_(pipelines::integration::ScalableTSDFVolume(
			voxel_length_,
			sdf_trunc_,
			pipelines::integration::TSDFVolumeColorType::RGB8)) {}

	// Integrate one RGB-D frame into TSDF volume (camera-to-world pose)
	void IntegrateFrame(
		const geometry::Image& depth,
		const geometry::Image& color,
		const camera::PinholeCameraIntrinsic& intrinsic,
		const Eigen::Matrix4d& extrinsic) {
		lock_guard<mutex> lock(mutex_);
		auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(
			color, depth, 1e3, 3.0, false);
		tsdf_.Integrate(*rgbd, intrinsic, extrinsic.inverse());
		integrated_frames_++;
	}

	// Extract current mesh snapshot from TSDF volume
	shared_ptr<geometry::TriangleMesh> ExtractMesh() {
		lock_guard<mutex> lock(mutex_);
		auto mesh = tsdf_.ExtractTriangleMesh();
		mesh->ComputeVertexNormals();
		return mesh;
	}

	// Save mesh to file
	void SaveMesh(const shared_ptr<geometry::TriangleMesh>& mesh,
		const string& filename) {
		io::WriteTriangleMesh(filename, *mesh);
	}

	size_t IntegratedFrameCount() const {
		lock_guard<mutex> lock(mutex_);
		return integrated_frames_;
	}

	shared_ptr<open3d::geometry::Image> MatToO3DImage(const cv::Mat& mat)
	{
		auto o3d_img = make_shared<geometry::Image>();

		o3d_img->width_ = mat.cols;
		o3d_img->height_ = mat.rows;
		o3d_img->num_of_channels_ = mat.channels();
		o3d_img->bytes_per_channel_ = static_cast<int>(mat.elemSize1());

		size_t total_bytes = mat.total() * mat.elemSize();
		o3d_img->data_.resize(total_bytes);
		memcpy(o3d_img->data_.data(), mat.data, total_bytes);

		return o3d_img;
	}

private:
	double voxel_length_;  // in meters
	double sdf_trunc_;     // truncation distance
	pipelines::integration::ScalableTSDFVolume tsdf_;
	mutable mutex mutex_;
	size_t integrated_frames_ = 0;
};

