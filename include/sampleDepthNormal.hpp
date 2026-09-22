#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace ORB_SLAM3 {

	std::vector<Eigen::Vector2d> gen2dSamplePoints(const cv::RotatedRect& ellipse)
	{
		std::vector<Eigen::Vector2d> subpixel_ellipse_points;

		double a = ellipse.size.width / 2.0;
		double b = ellipse.size.height / 2.0;
		double angle_rad = ellipse.angle * CV_PI / 180.0;

		double cos_a = std::cos(angle_rad);
		double sin_a = std::sin(angle_rad);

		const int Ksamples = 16;
		for (int k = 0; k < Ksamples; k++) {
			// t 是参数方程中的角度，从 0 到 2*PI
			double t = 2.0 * CV_PI * k / Ksamples;

			double cos_t = std::cos(t);
			double sin_t = std::sin(t);

			double x = ellipse.center.x + a * cos_t * cos_a - b * sin_t * sin_a;
			double y = ellipse.center.y + a * cos_t * sin_a + b * sin_t * cos_a;

			subpixel_ellipse_points.push_back(Eigen::Vector2d(x, y));
		}
		return subpixel_ellipse_points;
	}

	cv::Point2f reproject_ideal_to_distorted(
		const cv::Point2f& p_ideal_center,
		const cv::Mat& K,
		const cv::Mat& D)
	{
		// ====================================================
		// 1. 提取内参 (K) - 从 float 读取并转换为 double
		// ====================================================
		// 确保使用 .at<float> 读取，否则会导致内存解释错误
		const double fx = static_cast<double>(K.at<float>(0, 0));
		const double fy = static_cast<double>(K.at<float>(1, 1));
		const double cx = static_cast<double>(K.at<float>(0, 2));
		const double cy = static_cast<double>(K.at<float>(1, 2));

		// ====================================================
		// 2. 理想像素坐标 -> 归一化坐标 (xn, yn) [Double 计算]
		// ====================================================
		// 对应 Z=1 平面
		double xn = (static_cast<double>(p_ideal_center.x) - cx) / fx;
		double yn = (static_cast<double>(p_ideal_center.y) - cy) / fy;

		// ====================================================
		// 3. 应用正向畸变模型 [Double 计算]
		// ====================================================

		// 3.1. 提取畸变系数 (float -> double)
		// 假定 D 是连续存储的 (无论是 1xN 还是 Nx1)
		const double k1 = static_cast<double>(D.at<float>(0));
		const double k2 = static_cast<double>(D.at<float>(1));
		const double p1 = static_cast<double>(D.at<float>(2));
		const double p2 = static_cast<double>(D.at<float>(3));

		// 检查是否有 k3 (总元素个数 >= 5)
		const double k3 = (D.total() >= 5) ? static_cast<double>(D.at<float>(4)) : 0.0;

		// 3.2. 计算径向距离平方
		double r_sq = xn * xn + yn * yn;

		// 3.3. 计算径向畸变因子
		double radial_factor = 1.0 + k1 * r_sq + k2 * r_sq * r_sq + k3 * r_sq * r_sq * r_sq;

		// 3.4. 计算各项畸变分量
		double x_radial = xn * radial_factor;
		double y_radial = yn * radial_factor;

		double x_tan = 2.0 * p1 * xn * yn + p2 * (r_sq + 2.0 * xn * xn);
		double y_tan = p1 * (r_sq + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;

		// 3.5. 组合得到畸变后的归一化坐标 (xd, yd)
		double xd = x_radial + x_tan;
		double yd = y_radial + y_tan;

		// ====================================================
		// 4. 畸变归一化坐标 -> 畸变像素坐标 [转回 float 输出]
		// ====================================================
		float u_distorted = static_cast<float>(xd * fx + cx);
		float v_distorted = static_cast<float>(yd * fy + cy);

		return cv::Point2f(u_distorted, v_distorted);
	}

	float sampleBilinear(const cv::Mat& img, const cv::Point2f& p) {
		// img: CV_32FC1
		float x = p.x;
		float y = p.y;

		// 左上整数坐标
		int x0 = static_cast<int>(std::floor(x));
		int y0 = static_cast<int>(std::floor(y));
		int x1 = x0 + 1;
		int y1 = y0 + 1;

		// 边界检查（保证 x0,x1,y0,y1 都在图像范围内）
		if (x0 < 0 || x1 >= img.cols || y0 < 0 || y1 >= img.rows) {
			// 可以选择 clamp / 返回0 / 做更复杂的边界处理
			return 0.0f;
		}

		float dx = x - x0;
		float dy = y - y0;

		float I00 = img.at<float>(y0, x0);
		float I10 = img.at<float>(y0, x1);
		float I01 = img.at<float>(y1, x0);
		float I11 = img.at<float>(y1, x1);

		float I0 = I00 * (1.0f - dx) + I10 * dx;   // 上边线性插值
		float I1 = I01 * (1.0f - dx) + I11 * dx;   // 下边线性插值
		float I = I0 * (1.0f - dy) + I1 * dy;   // 垂直插值

		return I;
	}

	cv::Vec3f sampleNormalInEllipse(/*const cv::Mat& gray, */const cv::Mat& img, const cv::RotatedRect& e, int margin = 10) {
		CV_Assert(img.type() == CV_32FC3);

		// 椭圆的外接矩形
		cv::Rect bbox = e.boundingRect();

		// 向外扩 margin 像素（四个方向各 margin）
		bbox.x -= margin;
		bbox.y -= margin;
		bbox.width += 2 * margin;
		bbox.height += 2 * margin;

		// 和图像范围取交集，避免越界
		cv::Rect imgRect(0, 0, img.cols, img.rows);
		bbox = bbox & imgRect;
		if (bbox.empty()) {
			return cv::Vec3f::zeros();
		}
		//cv::Mat visualize, texture;
		//img.copyTo(visualize);
		//gray.copyTo(texture);
		//cv::rectangle(visualize, bbox, cv::Scalar(0, 0, 255.0), 2, cv::LINE_AA);
		//cv::rectangle(texture, bbox, cv::Scalar(0, 0, 255.0), 2, cv::LINE_AA);

		cv::Vec3f sumN(0.0f, 0.0f, 0.0f);
		int count = 0, count_valid = 0;

		// 遍历扩展后的外接矩形内所有像素
		for (int y = bbox.y; y < bbox.y + bbox.height; ++y) {
			for (int x = bbox.x; x < bbox.x + bbox.width; ++x) {
				const cv::Vec3f& n = img.at<cv::Vec3f>(y, x);
				sumN += n;
				++count;
				count_valid += cv::norm(n) > 1e-6;
			}
		}

		if (count == 0) {
			return cv::Vec3f::zeros();
		}
		if ((count_valid * 1.0 / count) < 0.9) {
			return cv::Vec3f::zeros();
		}
		//cv::Point2f textPos = e.center + cv::Point2f(10, 10);
		//cv::putText(visualize, to_string((count_valid * 1.0 / count)), textPos, cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
		//cv::putText(texture, to_string((count_valid * 1.0 / count)), textPos, cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
		//cv::imshow("visualize", visualize);
		//cv::imshow("texture", texture);
		//cv::waitKey(0);

		// 求平均
		cv::Vec3f N = sumN / static_cast<float>(count);

		// 归一化为单位法向量
		float norm = cv::norm(N);
		if (norm > 1e-6f) {
			N /= norm;
		}
		else {
			N.zeros();
		}

		return N;
	}
}
