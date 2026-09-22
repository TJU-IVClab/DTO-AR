#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <stdio.h>

using namespace std;

inline cv::Mat undisortDepthNormal(const cv::Mat& img, const cv::Mat& map1, const cv::Mat& map2)
{
	cv::Mat img_un_valid;
	img_un_valid.create(img.size(), img.type());

	if (img.type() == CV_32FC1)
	{
		cv::Mat img_un;
		cv::remap(img, img_un, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
		cv::Mat mask, mask_un;
		// 原始 depth_raw 中深度为 0 或 nan 视为无效
		mask = (img > 0);
		cv::remap(mask, mask_un, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
		mask_un /= 255;  // 归一化到 0/1

		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
		cv::Mat eroded_mask;
		cv::erode(mask_un,        // 输入掩码
			eroded_mask,                   // 输出腐蚀后掩码
			kernel,                        // 4×4 结构元
			cv::Point(-1, -1),                 // 锚点，(-1,-1) 表示结构元中心
			1,                             // 迭代次数：1
			cv::BORDER_CONSTANT,               // 边界模式，用常数填充
			cv::Scalar(0));                    // 边界常数填充值（0） :contentReference[oaicite:1]{index=1}

		eroded_mask.convertTo(eroded_mask, CV_32FC1, 1.0);
		img_un_valid = img_un.mul(eroded_mask);
	}
	else if (img.type() == CV_32FC3)
	{
		// No need to calculate eroded_mask, since invalid depth pixels have been culled.
		cv::Mat img_un;
		cv::remap(img, img_un, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
		// Normalize it!
		for (int y = 0; y < img_un.rows; ++y)
		{
			// 指针加速访问
			const cv::Vec3f* ptr_in = img_un.ptr<cv::Vec3f>(y);
			cv::Vec3f* ptr_out = img_un_valid.ptr<cv::Vec3f>(y);
			for (int x = 0; x < img_un.cols; ++x) {
				cv::Vec3f v = ptr_in[x];
				float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
				if (norm > 1e-6f) {
					ptr_out[x] = v / norm;
				}
				else {
					ptr_out[x] = cv::Vec3f(0, 0, 0);
				}
			}
		}
	}
	return img_un_valid;
}

inline void LoadImages(const string& strAssociationFilename, vector<string>& vstrImageFilenamesRGB,
	vector<string>& vstrImageFilenamesD, vector<string>& vstrImageFilenamesN, vector<double>& vTimestamps)
{
	ifstream fAssociation;
	fAssociation.open(strAssociationFilename.c_str());
	while (!fAssociation.eof())
	{
		string s;
		getline(fAssociation, s);
		if (!s.empty())
		{
			stringstream ss;
			ss << s;
			double t;
			string sRGB, sD, sN;
			ss >> t;
			vTimestamps.push_back(t);
			ss >> sRGB;
			vstrImageFilenamesRGB.push_back(sRGB);
			ss >> t;
			ss >> sD;
			vstrImageFilenamesD.push_back(sD);
			ss >> sN;
			vstrImageFilenamesN.push_back(sN);
		}
	}
}
inline bool loadRGBDNormal(const string& pathToSrc, const cv::Mat& map1, const cv::Mat& map2, vector<cv::Mat>& imGs, vector<cv::Mat>& imDs, vector<cv::Mat>& imNs)
{
	// Retrieve paths to images
	vector<string> vstrImageFilenamesRGB;
	vector<string> vstrImageFilenamesD;
	vector<string> vstrImageFilenamesN;
	vector<double> vTimestamps;
	LoadImages(pathToSrc + "/association.txt", vstrImageFilenamesRGB, vstrImageFilenamesD, vstrImageFilenamesN, vTimestamps);

	// Check consistency in the number of images and depthmaps
	int nImages = vstrImageFilenamesRGB.size();
	if (vstrImageFilenamesRGB.empty())
	{
		cerr << endl << "No images found in provided path." << endl;
		return false;
	}
	else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
	{
		cerr << endl << "Different number of images for rgb and depth." << endl;
		return false;
	}

	// Vector for tracking time statistics
	vector<float> vTimesTrack;
	vTimesTrack.resize(nImages);

	cout << endl << "-------" << endl;
	cout << "Images in the sequence: " << nImages << endl << endl;

	for (int i = 0; i < nImages; i++) {
		vector<cv::Mat> imNholder;
		cv::Mat imNx, imNy, imNz, imG, imD, imN;
		cv::Mat imG_un, imD_un, imN_un;
		double tframe;
		imG = cv::imread(pathToSrc + "/i_u8/" + vstrImageFilenamesRGB[i], cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
		//imG.convertTo(imGi, CV_8UC1, 4.0);
		cv::remap(imG, imG_un, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
		imGs.emplace_back(imG_un);
		imD = cv::imread(pathToSrc + "/d/" + vstrImageFilenamesD[i], cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
		//imD.convertTo(imDi, CV_32FC1, 1e-3);
		imD_un = undisortDepthNormal(imD, map1, map2);
		imDs.emplace_back(imD_un);
		imNx = cv::imread(pathToSrc + "/n/" + vstrImageFilenamesN[i], cv::IMREAD_UNCHANGED);
		imNy = cv::imread(pathToSrc + "/n/" + vstrImageFilenamesN[i].replace(vstrImageFilenamesN[i].find("X"), 1, "Y"), cv::IMREAD_UNCHANGED);
		imNz = cv::imread(pathToSrc + "/n/" + vstrImageFilenamesN[i].replace(vstrImageFilenamesN[i].find("Y"), 1, "Z"), cv::IMREAD_UNCHANGED);
		imNholder.push_back(imNx);
		imNholder.push_back(imNy);
		imNholder.push_back(imNz);
		cv::merge(imNholder, imN);
		imN_un = undisortDepthNormal(imN, map1, map2);
		imNs.emplace_back(imN_un);
		tframe = vTimestamps[i];
	}

	return true;
}

// 读取 timestamp tx ty tz qx qy qz qw
inline bool loadCameraPoses(const string& filename, vector<Eigen::Matrix4f>& poses)
{
	ifstream ifs(filename);
	if (!ifs.is_open()) {
		cerr << "couldn't open file: " << filename << endl;
		return false;
	}

	string line;
	while (getline(ifs, line)) {
		if (line.empty() || line[0] == '#')  // 跳过空行或注释
			continue;

		istringstream iss(line);
		double timestamp;
		double tx, ty, tz;
		double qx, qy, qz, qw;
		if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
			cerr << "wrong format, passing line: " << line << endl;
			continue;
		}

		// 构造 Eigen 对象（使用 float 类型）
		Eigen::Vector3f t(static_cast<float>(tx),
			static_cast<float>(ty),
			static_cast<float>(tz));
		Eigen::Quaternionf q(static_cast<float>(qw),
			static_cast<float>(qx),
			static_cast<float>(qy),
			static_cast<float>(qz));
		q.normalize();  // 确保四元数归一化

		// 构造 4×4 齐次矩阵
		Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
		T.block<3, 3>(0, 0) = q.toRotationMatrix();
		T.block<3, 1>(0, 3) = t * 1e3;

		poses.push_back(T);
	}

	return true;
}

inline void saveModeltoPly(vector<cv::Point3f>& points, vector<cv::Vec3b>& colors) {
	std::cout << "Model: total vertices → " << points.size() << std::endl;
	std::cout << "Saving geometry, this might take a while..." << std::endl;

	auto now = std::chrono::system_clock::now();
	time_t now_time_t = std::chrono::system_clock::to_time_t(now);
	tm now_tm = *localtime(&now_time_t);
	char buffer[100];
	strftime(buffer, sizeof(buffer), "%m_%d_%H-%M-%S", &now_tm);
	std::string time_str(buffer);

	std::string filename = "model_" + time_str + ".ply";
	std::ofstream outFile(filename, std::ios::binary);
	if (!outFile.is_open()) {
		std::cerr << "Error opening file for writing: " << filename << std::endl;
		return;
	}
	try {
		// ply header
		outFile << "ply\n";
		outFile << "format binary_little_endian 1.0\n";  // 使用二进制小端格式
		outFile << "element vertex " << points.size() << "\n";
		outFile << "property float x\n";
		outFile << "property float y\n";
		outFile << "property float z\n";
		outFile << "property uchar red\n";
		outFile << "property uchar green\n";
		outFile << "property uchar blue\n";
		outFile << "end_header\n";

		for (size_t i = 0; i < points.size(); ++i) {
			const cv::Point3f& point = (points)[i];
			const cv::Vec3b& color = (colors)[i];
			outFile.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
			outFile.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
			outFile.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
			outFile.write(reinterpret_cast<const char*>(&color[0]), sizeof(unsigned char));
			outFile.write(reinterpret_cast<const char*>(&color[1]), sizeof(unsigned char));
			outFile.write(reinterpret_cast<const char*>(&color[2]), sizeof(unsigned char));
		}
		outFile.close();
		std::cout << "Point cloud with colors saved to " << filename << std::endl;
	}
	catch (std::runtime_error& InternalException) {
		std::cout << std::endl
			<< "Exception was thrown: " << InternalException.what()
			<< std::endl;
	}
};


inline void textureDepthTo3D(cv::Mat& grey, cv::Mat& depth, Eigen::Matrix4f& Twc, vector<cv::Point3f>& points, vector<cv::Vec3b>& colors)
{
	const static double fx = 1154.93, fy = 1154.95, cx = 560.74, cy = 400.553;
	cv::Mat texture, texture_rgb;
	if (grey.type() == CV_32FC1) grey.convertTo(texture, CV_8UC1, 4.0);
	else if (grey.type() == CV_8UC1) texture = grey.clone();
	cv::cvtColor(texture, texture_rgb, cv::COLOR_GRAY2RGB);
	int w = grey.cols;
	int h = grey.rows;
	for (int v = 0; v < h; ++v)
		for (int u = 0; u < w; ++u) {
			const float xn = (u - cx) / fx;
			const float yn = (v - cy) / fy;
			const float& Z = depth.at<float>(v, u);
			Eigen::Vector4f p = Eigen::Vector4f(xn * Z, yn * Z, Z, 1.0);
			Eigen::Vector4f pt = Twc * p;
			points.push_back(cv::Point3f(pt.x(), pt.y(), pt.z()) * 1e-3); // unit: m

			const cv::Vec3b& color = texture_rgb.at<cv::Vec3b>(v, u);
			colors.push_back(color);
		}
}

struct MatteSpot {
	cv::RotatedRect ellipse;
	cv::Point2f centerUn; // undistorted ellipse center
	cv::Point3d p;
	cv::Point3d n;
	int count;
	int ID;
	MatteSpot(cv::RotatedRect e) : ellipse(e), centerUn(0, 0), p(0, 0, 0), n(0, 0, 0), count(1), ID(-1) {};
};
inline cv::Mat ExtractMatteSpots(const cv::Mat& im)
{
	vector<MatteSpot> MS;
	cv::Mat imGray;
	im.convertTo(imGray, CV_8UC1, 255.0 / 512);

	// Apply Gaussian blur to reduce noise
	cv::Mat blurred;
	cv::GaussianBlur(imGray, blurred, cv::Size(5, 5), 0);

	// Apply Canny edge detection
	cv::Mat edges;
	cv::Canny(blurred, edges, 25, 80, 3, true);
	//cv::imshow("edge", edges);

	// Ellipse fitting
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	vector<cv::RotatedRect> ellipses;
	for (size_t i = 0; i < contours.size(); i++)
		if (contours[i].size() >= 15 /*&& hierarchy[i][2] == -1 && hierarchy[i][3] != -1*/)
		{
			cv::RotatedRect e = cv::fitEllipseDirect(contours[i]);
			// d(pixel):[5(pi*2.5^2≈20), 21(pi*10.5^2≈350)], 
			//float area = (CV_PI * e.size.width * e.size.height / 4.0) / (imGray.size().width == 1680 ? 2.25 : 1.0);
			//float ratio = e.size.width / e.size.height;
			//if (area > 20 && area < 750 && abs(ratio - 1) < 0.5) { // Check validation
			ellipses.emplace_back(e);
			//cout << ellipse.center.x  << "-" << area << endl;
		//}
		}

	// Work with concentric ellipses, MatteSpot{inner⚪} has a true diameter of 10mm, {outer⚪} has a true diameter of 20mm
	float thres_1 = 2.0f; // pixel => Corresponding to the same Matte Spot 🧿
	float thres_2 = 5e-2f; // pixel => Corresponding to the same actual contour 

	for (const cv::RotatedRect& e : ellipses)
	{
		bool to_merged = false;
		const float e_area = e.boundingRect().area();

		for (MatteSpot& mergedCenter : MS)
		{
			cv::RotatedRect& me = mergedCenter.ellipse;
			double distance = cv::norm(e.center - me.center);
			const float me_area = me.boundingRect().area();

			if (distance < thres_1) // merge concentric ellipses，and take the center of the larger inner ellipse
			{
				to_merged = true;

				if ((2.5f * e_area) < me_area) { me = e; }
				else if (e_area > me_area && e_area < 2.5f * me_area) { me = e; }

				mergedCenter.count++;
				break;
			}
		}
		if (!to_merged)
			MS.emplace_back(MatteSpot(e));
	}

	cv::Mat imGray_filtered;
	cv::cvtColor(imGray, imGray_filtered, cv::COLOR_GRAY2RGB);

	// Restore the 3D coordinates and normal vectors
	for (auto iter = MS.begin(); iter != MS.end();)
	{
		if (iter->count <= 2) {
			iter = MS.erase(iter); //返回下一个有效的迭代器，无需+1 
			continue;
		}
		float ratio = iter->ellipse.size.width / iter->ellipse.size.height;
		float area = (CV_PI * iter->ellipse.size.width * iter->ellipse.size.height / 4.0) / (imGray.size().width == 1680 ? 2.25 : 1.0);
		if (abs(ratio - 1) >= 0.5 || area <= 20 || area >= 350) {
			iter = MS.erase(iter); //返回下一个有效的迭代器，无需+1 
			continue;
		}

		const float& u = iter->ellipse.center.x;
		const float& v = iter->ellipse.center.y;

		vector<cv::Point2f> pts; // _bottomLeft_, _topLeft_, topRight, bottomRight.
		iter->ellipse.points(pts);


		//uchar Ic = imGray.at<uchar>(int(v), int(u));
		uint8_t Ie = 0;
		bool is_valid = true;
		for (cv::Point2f& pt : pts) {
			cv::circle(imGray_filtered, pt, 2, cv::Scalar(255, 255, 0), 1);
			const float& ptx = round(pt.x);
			const float& pty = round(pt.y);
			Ie = imGray.at<uint8_t>(pty, ptx);
			//int diff = Ic - Ie;
			//cout << "!!!" << ptx << " :: " << pty << "::" << int(Ie) << endl;
			if (Ie > 50) is_valid = false;
		}
		//if (!is_valid) {
		//	iter = MS.erase(iter);
		//	continue;
		//}
		/*int rapidChangePixelCount = {};
		for (int r = 1; r <= 15; ++r)
			if (++um < imGray.size().width) {
				Ie = imGray.at<uchar>(vm, um);
				if ((Ic - Ie) > 50) {
					rapidChangePixelCount = r;
					break;
				}
			}
		if (!rapidChangePixelCount) {
			iter = MS.erase(iter);
			continue;
		}
		iter->count = rapidChangePixelCount;*/
		//cv::imwrite("test.png", imGray_filtered);
		cv::ellipse(imGray_filtered, iter->ellipse, is_valid ? cv::Scalar(0, 255.0, 0) : cv::Scalar(0, 0, 255.0), cv::FILLED);
		//for (cv::RotatedRect& e : iter->mve)
		//	cout << e.center << " :: " << e.boundingRect().area() << " | ";
		/*cout << " u: " << u << ", v: " << v << " un: " << centerUn.x << ", vn: " << centerUn.y
			<< ", x: " << iter->p.x << ", y: " << iter->p.y << ", z: " << iter->p.z
			<< ", nx: " << iter->n.x << ", ny: " << iter->n.y << ", nz: " << iter->n.z
			 << ", count: " << iter->count << ", radius:" << rapidChangePixelCount << endl;*/

		++iter;
	}
	/*cv::Mat imGray_filtered = cv::Mat::zeros(imGray.size(), CV_32FC1);
	//imGray_filtered.convertTo(imGray_filtered, CV_8UC3);
	for (size_t i = 0; i < MS.size(); i++) {
		MatteSpot& mergedCenter = MS[i];
		//cv::ellipse(imGray_filtered, mergedCenter.ellipse, cv::Scalar(255, 255, 255), 12);
		cv::ellipse(imGray_filtered, mergedCenter.ellipse, cv::Scalar(255.0), cv::FILLED);
	}
	//imGray_filtered.convertTo(imGray_filtered, CV_32FC1, 1.0 / 255);
	//cv::imshow("aaa", imGray_filtered);
	//cv::imwrite("12345.tif", imGray_filtered);
	//cv::waitKey(0);*/
	return imGray_filtered;
}