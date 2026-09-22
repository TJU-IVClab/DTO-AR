#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <memory>
#include <map>

#include "Settings.h"
#include "PhoXi.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <osg/ref_ptr>
#include <osg/Node>
#include <osg/Node>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>

#if defined(_WIN32)
#define DELIMITER "\\"
#elif defined(__linux__) || defined(__APPLE__)
#define DELIMITER "/"
#endif
#define LOCAL_CROSS_SLEEP(Millis) this_thread::sleep_for(chrono::milliseconds(Millis));

using namespace std;

/*class pcdHandler {
public:
	pcdHandler()
		: frameIndex_KF(-1), Twc_KF(Eigen::Matrix4f::Identity()),
		pcdPoints_KF(make_shared<vector<cv::Point3f>>()),
		pcdColors_KF(make_shared<vector<cv::Vec3b>>()) {}

	// copying to the class's internal storage
	void setData(size_t index, Eigen::Matrix4f Twc, vector<cv::Point3f> points, vector<cv::Vec3b> colors) {
		unique_lock<mutex> lock(mutex_pcd);
		frameIndex_KF = index;
		Twc_KF = Twc;
		*pcdPoints_KF = points; // *ptr = data v.s. ptr = &data
		*pcdColors_KF = colors;
	}
	// Avoid copying and return read-only references
	void getData(size_t& index, Eigen::Matrix4f& Twc, shared_ptr<const vector<cv::Point3f>>& points, shared_ptr<const vector<cv::Vec3b>>& colors) {
		unique_lock<mutex> lock(mutex_pcd);
		index = frameIndex_KF;
		Twc = Twc_KF;
		points = pcdPoints_KF;
		colors = pcdColors_KF;
	}

private:
	size_t frameIndex_KF;
	Eigen::Matrix4f Twc_KF;
	shared_ptr<vector<cv::Point3f>> pcdPoints_KF;
	shared_ptr<vector<cv::Vec3b>> pcdColors_KF;

protected:
	mutex mutex_pcd;
};*/
class pcdHandler
{
public:
	pcdHandler()
		: frameIndex_KF(-1), Twc_KF(Eigen::Matrix4f::Identity()),
		depth_KF(make_unique<cv::Mat>()), color_KF(make_unique<cv::Mat>())
	{
	}

	// copying to the class's internal storage
	void setData(size_t index, Eigen::Matrix4f Twc, cv::Mat depth, cv::Mat color)
	{
		unique_lock<mutex> lock(mutex_pcd);
		frameIndex_KF = index;
		Twc_KF = Twc;
		*depth_KF = depth; // *ptr = data v.s. ptr = &data
		*color_KF = color;
	}
	//// Avoid copying and return read-only references
	//void getData(size_t& index, Eigen::Matrix4f& Twc, unique_ptr<const cv::Mat>& depth, unique_ptr<const cv::Mat>& color)
	//{
	//	unique_lock<mutex> lock(mutex_pcd);
	//	index = frameIndex_KF;
	//	Twc = Twc_KF;
	//	depth = depth_KF;
	//	color = color_KF;
	//}

//private:
	size_t frameIndex_KF;
	Eigen::Matrix4f Twc_KF;
	unique_ptr<cv::Mat> depth_KF;
	unique_ptr<cv::Mat> color_KF;

//protected:
	mutex mutex_pcd;
};

//typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
//typedef K::Point_3 Point;
//typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
//typedef CGAL::Surface_mesh<Point> SurfaceMesh;
//class IncrementalMeshBuilder {
//public:
//	IncrementalMeshBuilder() {}
//
//	// 添加新点到点集合并增量更新网格
//	void add_points(const std::vector<Point>& new_points) {
//		points.insert(points.end(), new_points.begin(), new_points.end());
//		update_mesh(new_points);
//	}
//
//	// 导出网格到文件
//	void export_mesh(const std::string& filename) {
//		if (!mesh.is_empty()) {
//			std::ofstream output(filename);
//			if (output) {
//				output << mesh;
//				std::cout << "网格已保存到 " << filename << std::endl;
//			}
//			else {
//				std::cerr << "无法写入文件：" << filename << std::endl;
//			}
//		}
//		else {
//			std::cerr << "当前没有可导出的网格。" << std::endl;
//		}
//	}
//
//private:
//	std::vector<Point> points;      // 存储所有点
//	SurfaceMesh mesh;              // 存储生成的网格
//
//	// 更新网格
//	void update_mesh(const std::vector<Point>& new_points) {
//		if (points.size() < 4) {
//			std::cerr << "点数不足，无法生成网格。" << std::endl;
//			return;
//		}
//
//		// 使用 Delaunay 三角剖分生成局部网格
//		Delaunay dt;
//		dt.insert(new_points.begin(), new_points.end());
//
//		// 将 Delaunay 三角剖分的单元加入 SurfaceMesh
//		SurfaceMesh new_mesh;
//		for (auto it = dt.finite_cells_begin(); it != dt.finite_cells_end(); ++it) {
//			auto v0 = new_mesh.add_vertex(it->vertex(0)->point());
//			auto v1 = new_mesh.add_vertex(it->vertex(1)->point());
//			auto v2 = new_mesh.add_vertex(it->vertex(2)->point());
//			auto v3 = new_mesh.add_vertex(it->vertex(3)->point());
//
//			// 创建面
//			new_mesh.add_face(v0, v1, v2);
//			new_mesh.add_face(v0, v1, v3);
//			new_mesh.add_face(v0, v2, v3);
//			new_mesh.add_face(v1, v2, v3);
//		}
//
//		// 合并新网格到现有网格
//		merge_mesh(mesh, new_mesh);
//	}
//
//	// 合并两个网格
//	void merge_mesh(SurfaceMesh& base_mesh, const SurfaceMesh& new_mesh) {
//		std::map<Point, SurfaceMesh::Vertex_index> point_map;
//
//		// 将 base_mesh 的顶点加入映射
//		for (auto v : base_mesh.vertices()) {
//			point_map[base_mesh.point(v)] = v;
//		}
//
//		// 添加 new_mesh 的顶点到 base_mesh
//		for (auto v : new_mesh.vertices()) {
//			Point p = new_mesh.point(v);
//
//			// 如果点已存在，则跳过
//			if (point_map.find(p) == point_map.end()) {
//				auto new_vertex = base_mesh.add_vertex(p);
//				point_map[p] = new_vertex;
//			}
//		}
//
//		// 添加 new_mesh 的面到 base_mesh
//		for (auto f : new_mesh.faces()) {
//			std::vector<SurfaceMesh::Vertex_index> face_vertices;
//			for (auto h : CGAL::halfedges_around_face(new_mesh.halfedge(f), new_mesh)) {
//				face_vertices.push_back(point_map[new_mesh.point(new_mesh.target(h))]);
//			}
//			if (face_vertices.size() == 3) {
//				base_mesh.add_face(face_vertices[0], face_vertices[1], face_vertices[2]);
//			}
//		}
//
//		// 修复边界和拓扑
//		CGAL::Polygon_mesh_processing::stitch_borders(base_mesh);
//	}
//};


class MeshUpdate : public osg::NodeCallback {
public:
	MeshUpdate() { mGeode = new osg::Geode; }

	virtual void operator()(osg::Node* node_, osg::NodeVisitor* nv) {
		osg::Group* node = dynamic_cast<osg::Group*>(node_);
		int numGroupChildren = node->getNumChildren();
		if (numGroupChildren != 0) {
			node->removeChildren(0, numGroupChildren);
		}
		{
			unique_lock<mutex> lock(mutex_mesh);
			node->addChild(mGeode);
		}
		traverse(node, nv);
	}

	void setMesh(osg::ref_ptr<osg::Geometry> mGeometry) {
		unique_lock<mutex> lock(mutex_mesh);
		mGeode = new osg::Geode;
		mGeode->addChild(mGeometry);
	}

protected:
	osg::ref_ptr<osg::Geode> mGeode;
	mutex mutex_mesh;
};

namespace ORB_SLAM3
{
	//class Atlas;
	//class Tracking;
	//class LocalMapping;

	struct FrameData {
		size_t frameIndex;
		double frameTimestamp;
		cv::Mat t_u8;
		cv::Mat d_f32;
		cv::Mat n_f32;
		cv::Mat c_f32;
		FrameData(int w, int h) : frameIndex(-1), frameTimestamp(0),
			t_u8(cv::Mat(h, w, CV_8UC1)),
			d_f32(cv::Mat(h, w, CV_32FC1)),
			n_f32(cv::Mat(h, w, CV_32FC3)),
			c_f32(cv::Mat(h, w, CV_32FC1)) {};
	};

	class PhotoneoSensor {
	public:
		// M+ Scanner / Camera mode
		const unsigned int imgWidth = 1680, imgHeight = 1200;
		double fx = 1732.39, fy = 1732.43, cx = 841.36, cy = 601.08;
		//const unsigned int imgWidth = 1120, imgHeight = 800;
		//double fx = 1154.93, fy = 1154.95, cx = 560.74, cy = 400.553;
		double k1 = -0.0867707, k2 = 0.147489, k3 = -0.0370996;
		double p1 = 0.000417617, p2 = 0.000554355;
		//// L
		//const unsigned int imgWidth = 2064, imgHeight = 1544;
		//double fx = 2335.69, fy = 2336.19, cx = 1058.4, cy = 773.34;
		//double k1 = -0.237187, k2 = 0.165352, k3 = -0.069309;
		//double p1 = 0.000197394, p2 = -0.000421582;
	
		int LEDPower = 4095;
		double FPS = 20.0;
		double Confidence = 1.0;
		cv::Mat mK, mDistCoeffs;
		// Normalized image plane coordinates after undistortion
		cv::Mat Raymap;
		cv::Mat map1, map2;

		PhotoneoSensor(Settings* settings);
		~PhotoneoSensor();
		std::string getSensorName() const {
			return "MotionCam-3D-CTR-095";
		}
		pho::api::PhoXiFactory Factory;
		pho::api::PPhoXi PhoXiDevice;
		vector<pho::api::PhoXiDeviceInformation> DeviceList;
		std::string OutputFolder = "";

		pho::api::PFrame SampleFrame;
		FrameData* mFrame = new FrameData(imgWidth, imgHeight);

		//list<KeyFrame*> mlNewKeyFrames;
		//KeyFrame* mpCurrentKeyFrame;
		bool mbFinishRequested = false;

		osg::ref_ptr<MeshUpdate> meshUpdateCb = new MeshUpdate;
		osg::ref_ptr<osg::Group> mGroup = new osg::Group;
		osg::ref_ptr<osgViewer::Viewer> viewer = new osgViewer::Viewer;
		// keyframe pcd data
		pcdHandler* mpcdHandler = new pcdHandler;
		// total pcd data
		osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array();
		osg::ref_ptr<osg::Vec3ubArray> colors = new osg::Vec3ubArray();
		osg::ref_ptr<osg::UIntArray> kfIdSF = new osg::UIntArray();
		osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array();

		void createFirstConnected();
		void CorrectDisconnectExample();
		void viewerRun();
		bool getFrame();
		void getFrame_offline(cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, double&, int&);
		void renderingPC();
		void saveModeltoPly(string path="");
		void RequestFinish();
		bool checkFinish();
		//void InsertKeyFrame(KeyFrame* pKF);
		//bool CheckNewKeyFrames();

		// evaluation: convert Intensity Depth Normal to .ply
		void IDNtoPLY(cv::Mat&, cv::Mat&, cv::Mat&, int frameIndex);

	private:
		void GetAvailableDevicesExample();
		void ConnectPhoXiDeviceBySerialExample();
		void FreerunExample();
		void DataHandlingExample();
		virtual void ChangeMotionCamExample();

		void PrintFrameInfo(const pho::api::PFrame& Frame);
		void PrintFrameData(const pho::api::PFrame& Frame);
		template <class T> bool ReadLine(T& Output) const {
			string Input;
			getline(cin, Input);
			stringstream InputSteam(Input);
			return (InputSteam >> Output) ? true : false;
		}
		bool ReadLine(std::string& Output) const {
			getline(cin, Output);
			return true;
		}

	protected:
		mutex mMutexFinish;
		//	mutex mMutexNewKF;
		//	LocalMapping* mpLocalMapper;
	};
	class Scanner : public PhotoneoSensor {
		//public:
		//	const unsigned int imgWidth = 1680, imgHeight = 1200;
		//	double fx = 1732.39;
		//	double fy = 1732.43;
		//	double cx = 841.36;
		//	double cy = 601.08;
		//	double focus = 12.1269;
		//	double cell_unit = 0.007;
		//	cv::Mat getCameraMatrix() override {
		//		return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
		//	}
		//	cv::Mat getDistCoeffs() override {
		//		return (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
		//	}
		//private:
		//	void ChangeMotionCamExample() override {
		//		auto& P = PhoXiDevice->MotionCam;
		//		P->LaserPower = 4095;
		//		P->LEDPower = 4095;
		//		P->MaximumFPS = 5.0;
		//		P->OperationMode = pho::api::PhoXiOperationMode::Camera;
		//
		//		auto& PCM = PhoXiDevice->MotionCamCameraMode;
		//		vector<double> SinglePatternExposures = PhoXiDevice->SupportedSinglePatternExposures;
		//		PCM->Exposure = SinglePatternExposures[0]; // 0 for 10.24
		//		PCM->OutputTopology = pho::api::PhoXiOutputTopology::FullGrid;
		//		PCM->CodingStrategy = pho::api::PhoXiCodingStrategy::Interreflections;
		//		PCM->TextureSource = pho::api::PhoXiTextureSource::LED;
		//
		//		auto& PCMPS = PhoXiDevice->ProcessingSettings;
		//		PCMPS->NormalsEstimationRadius = 2;
		//		PCMPS->Confidence = 1.0;// mm
		//		//Check
		//		if (!PCMPS.isLastOperationSuccessful())
		//			throw runtime_error(PCMPS.GetLastErrorMessage().c_str());
		//	};
	};
}


