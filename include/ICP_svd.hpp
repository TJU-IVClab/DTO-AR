#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace ORB_SLAM3 {
	struct MatteSpot;

	cv::Point3d calculateCentroid(const std::vector<cv::Point3d>& points) {
		double sumX = 0, sumY = 0, sumZ = 0;
		for (const auto& p : points) {
			sumX += p.x;
			sumY += p.y;
			sumZ += p.z;
		}
		int n = points.size();
		return cv::Point3d(sumX / n, sumY / n, sumZ / n);
	}
	double calculateTransform(std::vector<std::pair<cv::Point3d, Eigen::Vector3f>>& pairs, cv::Mat& transform) {
		std::vector<cv::Point3d> points1, points2;
		for (const auto& pair : pairs) {
			points1.push_back(pair.first);
			points2.push_back(cv::Point3d(pair.second.x(), pair.second.y(), pair.second.z()));
		}
		// 计算质心
		cv::Point3d centroid1 = calculateCentroid(points1);
		cv::Point3d centroid2 = calculateCentroid(points2);

		// 去中心化
		std::vector<cv::Point3d> centered1, centered2;
		for (size_t i = 0; i < points1.size(); ++i) {
			centered1.emplace_back(points1[i].x - centroid1.x, points1[i].y - centroid1.y, points1[i].z - centroid1.z);
			centered2.emplace_back(points2[i].x - centroid2.x, points2[i].y - centroid2.y, points2[i].z - centroid2.z);
		}

		// 计算协方差矩阵
		cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
		for (size_t i = 0; i < centered1.size(); ++i) {
			covariance += cv::Mat(centered2[i]) * cv::Mat(centered1[i]).t();
		}

		// 奇异值分解
		cv::Mat U, S, Vt;
		cv::SVDecomp(covariance, S, U, Vt);

		// 计算旋转矩阵
		cv::Mat R = U * Vt;
		if (cv::determinant(R) < 0) {
			R.col(2) *= -1;
		}

		// 计算平移向量
		cv::Mat t = (cv::Mat_<double>(3, 1) <<
			centroid2.x - (R.at<double>(0, 0) * centroid1.x + R.at<double>(0, 1) * centroid1.y + R.at<double>(0, 2) * centroid1.z),
			centroid2.y - (R.at<double>(1, 0) * centroid1.x + R.at<double>(1, 1) * centroid1.y + R.at<double>(1, 2) * centroid1.z),
			centroid2.z - (R.at<double>(2, 0) * centroid1.x + R.at<double>(2, 1) * centroid1.y + R.at<double>(2, 2) * centroid1.z));

		// 构建4x4变换矩阵
		transform = cv::Mat::eye(4, 4, CV_64F);
		R.copyTo(transform(cv::Rect(0, 0, 3, 3)));
		t.copyTo(transform(cv::Rect(3, 0, 1, 3)));

		// 计算重投影误差
		double repjErr = 0;
		for (size_t i = 0; i < points1.size(); ++i) {
			// 将点从点云1变换到点云2坐标系
			cv::Mat point_h = transform * (cv::Mat_<double>(4, 1) << points1[i].x, points1[i].y, points1[i].z, 1);

			// 提取变换后的点
			cv::Point3d transformed_point(point_h.at<double>(0), point_h.at<double>(1), point_h.at<double>(2));

			// 计算变换后点与目标点的距离
			repjErr += cv::norm(transformed_point - points2[i]);
		}
		repjErr /= points1.size(); // 计算平均误差

		return repjErr;
	}
}
