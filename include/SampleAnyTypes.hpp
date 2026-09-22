#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace ORB_SLAM3 {

	inline std::vector<cv::Point2f> gen2dSamplePoints(const cv::RotatedRect& ellipse)
	{
		std::vector<cv::Point2f> subpixel_ellipse_points;

		double a = ellipse.size.width / 2.0;
		double b = ellipse.size.height / 2.0;
		double angle_rad = ellipse.angle * CV_PI / 180.0;

		double cos_a = std::cos(angle_rad);
		double sin_a = std::sin(angle_rad);

		const int Ksamples = 64;
		for (int k = 0; k < Ksamples; k++) {
			// t 是参数方程中的角度，从 0 到 2*PI
			double t = 2.0 * CV_PI * k / Ksamples;

			double cos_t = std::cos(t);
			double sin_t = std::sin(t);

			double x = ellipse.center.x + a * cos_t * cos_a - b * sin_t * sin_a;
			double y = ellipse.center.y + a * cos_t * sin_a + b * sin_t * cos_a;

			subpixel_ellipse_points.push_back(cv::Point2f(x, y));
		}
		return subpixel_ellipse_points;
	}

	inline cv::Point2f reproject_ideal_to_distorted(
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

	inline float sampleBilinear(const cv::Mat& img, const cv::Point2f& p)
	{
		float I = .0f;
		try {
			// img: CV_32FC1
			float x = p.x;
			float y = p.y;

			// 左上整数坐标
			int x0 = static_cast<int>(std::floor(x));
			int y0 = static_cast<int>(std::floor(y));
			int x1 = x0 + 1;
			int y1 = y0 + 1;

			// boundary checks
			if (x0 < 0 || x1 >= img.cols || y0 < 0 || y1 >= img.rows) {
				return .0f;
			}

			float I00 = img.at<float>(y0, x0);
			float I10 = img.at<float>(y0, x1);
			float I01 = img.at<float>(y1, x0);
			float I11 = img.at<float>(y1, x1);

			if (I00 == .0f || I10 == .0f || I01 == .0f || I11 == .0f) {
				return .0f;
			}

			float dx = x - x0;
			float dy = y - y0;

			float I0 = I00 * (1.0f - dx) + I10 * dx;   // 上边线性插值
			float I1 = I01 * (1.0f - dx) + I11 * dx;   // 下边线性插值
			I = I0 * (1.0f - dy) + I1 * dy;   // 垂直插值
		}
		catch (cv::Exception e) {
			cout << e.what() << endl;
		}
		return I;
	}

	inline cv::Vec3f sampleNormalInEllipse(/*const cv::Mat& gray, */const cv::Mat& img,
		const cv::RotatedRect& e,
		int margin = 10)
	{
		CV_Assert(img.type() == CV_32FC3);

		// 1. 椭圆外接矩形
		cv::Rect bbox = e.boundingRect();

		// 2. 向外扩 margin 像素（四个方向各 margin）
		bbox.x -= margin;
		bbox.y -= margin;
		bbox.width += 2 * margin;
		bbox.height += 2 * margin;

		// 3. 和图像范围取交集，避免越界
		cv::Rect imgRect(0, 0, img.cols, img.rows);
		bbox = bbox & imgRect;
		if (bbox.empty()) {
			return cv::Vec3f::zeros();
		}

		// 4. 在 bbox 内部建立 mask，先全 0
		cv::Mat mask(bbox.height, bbox.width, CV_8UC1, cv::Scalar(0));

		// 把椭圆从全图坐标转换到 bbox 局部坐标
		cv::RotatedRect e_local = e;
		e_local.center.x -= static_cast<float>(bbox.x);
		e_local.center.y -= static_cast<float>(bbox.y);

		// 用实心椭圆填充 mask 中的椭圆区域
		// 这里用 center + axes + angle 的版本，厚度为 -1 表示填充
		cv::Size axes(cvRound(e_local.size.width * 0.5f),
			cvRound(e_local.size.height * 0.5f));
		if (axes.width > 0 && axes.height > 0) {
			cv::ellipse(mask,
				e_local.center,
				axes,
				e_local.angle,
				0.0, 360.0,
				cv::Scalar(255),
				-1,                // 填充
				cv::LINE_AA);
		}

		/*cv::Mat visualize, texture;
		img.copyTo(visualize);
		gray.copyTo(texture);
		cv::cvtColor(texture, texture, cv::COLOR_GRAY2BGR);
		cv::rectangle(visualize, bbox, cv::Scalar(0, 0, 255.0), 2, cv::LINE_AA);
		cv::rectangle(texture, bbox, cv::Scalar(0, 0, 255.0), 2, cv::LINE_AA);*/

		// 5. 对 “bbox 内的像素 - 椭圆区域（mask=255）” 的补集做统计
		cv::Vec3f sumN(0.0f, 0.0f, 0.0f);
		int count = 0, count_valid = 0;

		for (int y = bbox.y; y < bbox.y + bbox.height; ++y) {
			const uchar* mask_row = mask.ptr<uchar>(y - bbox.y);
			for (int x = bbox.x; x < bbox.x + bbox.width; ++x) {
				// 椭圆内部（mask=255）直接跳过，只取补集区域
				if (mask_row[x - bbox.x] != 0) {
					continue;
				}

				const cv::Vec3f& n = img.at<cv::Vec3f>(y, x);
				sumN += n;
				++count;
				if (cv::norm(n) > 1e-6f) {
					++count_valid;
				}
				//cv::circle(texture, cv::Point2f(x, y), 1, cv::Scalar(0, 255, 0), -1);
			}
		}

		// 6. 环形区域没有有效像素
		if (count == 0) {
			auto u = static_cast<int>(round(e.center.x));
			auto v = static_cast<int>(round(e.center.y));
			return img.at<cv::Vec3f>(v, u);
		}

		// 非零向量比例过低则认为无效，回退到原处采样
		if ((count_valid * 1.0 / count) < 0.9) {
			auto u = static_cast<int>(round(e.center.x));
			auto v = static_cast<int>(round(e.center.y));
			return img.at<cv::Vec3f>(v, u);
		}

		/*cv::Point2f textPos = e.center + cv::Point2f(10, 10);
		cv::putText(visualize, to_string((count_valid * 1.0 / count)), textPos, cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
		cv::putText(texture, to_string((count_valid * 1.0 / count)), textPos, cv::FONT_HERSHEY_TRIPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
		cv::imshow("visualize", visualize);
		cv::imshow("texture", texture);
		cv::waitKey(0);*/

		// 7. 求平均并归一化
		cv::Vec3f N = sumN / static_cast<float>(count);
		float normN = cv::norm(N);
		if (normN > 1e-6f) {
			N /= normN;
		}
		else {
			N = cv::Vec3f::zeros();
		}

		return N;
	}

	inline bool isValidDepth(float z) {
		return std::isfinite(z) && z > 1e-3f; // >0mm
	}

	// 计算 median
	inline float medianOf(std::vector<float>& v) {
		if (v.empty()) return 0.f;
		size_t mid = v.size() / 2;
		std::nth_element(v.begin(), v.begin() + mid, v.end());
		float m = v[mid];
		if (v.size() % 2 == 0) {
			std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
			m = 0.5f * (m + v[mid - 1]);
		}
		return m;
	}

	// 3D 平面：n·p + d = 0
	inline bool fitPlanePCA(const std::vector<cv::Vec3f>& pts, cv::Vec3f& n, float& d) {
		if (pts.size() < 20) return false;

		cv::Vec3d mean(0, 0, 0);
		for (auto& p : pts) mean += cv::Vec3d(p[0], p[1], p[2]);
		mean *= (1.0 / (double)pts.size());

		cv::Matx33d C(0, 0, 0, 0, 0, 0, 0, 0, 0);
		for (auto& p : pts) {
			cv::Vec3d q = cv::Vec3d(p[0], p[1], p[2]) - mean;
			C(0, 0) += q[0] * q[0]; C(0, 1) += q[0] * q[1]; C(0, 2) += q[0] * q[2];
			C(1, 0) += q[1] * q[0]; C(1, 1) += q[1] * q[1]; C(1, 2) += q[1] * q[2];
			C(2, 0) += q[2] * q[0]; C(2, 1) += q[2] * q[1]; C(2, 2) += q[2] * q[2];
		}

		cv::Mat eval, evec;
		cv::eigen(cv::Mat(C), eval, evec); // 特征值降序，最后一行是最小特征值对应向量
		cv::Vec3d nn(evec.at<double>(2, 0), evec.at<double>(2, 1), evec.at<double>(2, 2));
		double normv = std::sqrt(nn.dot(nn));
		if (normv < 1e-12) return false;
		nn *= (1.0 / normv);

		// 统一法向方向：让 n.z > 0（可选）
		if (nn[2] < 0) nn *= -1.0;

		n = cv::Vec3f((float)nn[0], (float)nn[1], (float)nn[2]);
		d = -(n[0] * (float)mean[0] + n[1] * (float)mean[1] + n[2] * (float)mean[2]);
		return true;
	}

	inline bool fitPlaneRobustMAD(std::vector<cv::Vec3f>& pts, cv::Vec3f& n, float& d) {
		if (!fitPlanePCA(pts, n, d)) return false;

		// 计算残差（点到平面带符号距离，未除以|n|因为|n|=1）
		std::vector<float> r; r.reserve(pts.size());
		for (auto& p : pts) r.push_back(n.dot(p) + d);

		// MAD
		std::vector<float> absr = r;
		for (auto& v : absr) v = std::fabs(v);
		float mad = medianOf(absr);
		if (mad < 1e-6f) mad = 1e-6f;

		// 阈值：k * MAD（k=3~5 常用）
		const float k = 4.0f;
		float th = k * mad;

		std::vector<cv::Vec3f> inliers;
		inliers.reserve(pts.size());
		for (size_t i = 0; i < pts.size(); ++i) {
			if (std::fabs(r[i]) <= th) inliers.push_back(pts[i]);
		}
		if (inliers.size() < 20) return true; // inlier 太少就用第一次结果
		return fitPlanePCA(inliers, n, d);
	}

	// 对单个 RotatedRect 做纠正
	inline bool correctOneMarker(cv::Mat& depth,
		const cv::RotatedRect& roi,
		const cv::Mat& K,
		const cv::Mat& D,
		int marginPx = 12,
		int minBandSamples = 200)
	{
		cv::Mat1f depthMm = depth;
		const int W = depthMm.cols, H = depthMm.rows;

		cv::RotatedRect outer = roi;
		outer.size.width += 2.0f * marginPx;
		outer.size.height += 2.0f * marginPx;

		cv::Rect bbox = outer.boundingRect();
		bbox &= cv::Rect(0, 0, W, H);
		if (bbox.width <= 2 || bbox.height <= 2) return false;

		// 在 bbox 局部生成 inner/outer mask
		cv::Mat1b innerMask(bbox.height, bbox.width, uchar(0));
		cv::Mat1b outerMask(bbox.height, bbox.width, uchar(0));

		auto fillEllipseFromRotRect = [&](const cv::RotatedRect& rr,
			cv::Mat1b& mask,
			float axisScale = 1.0f) {
				// rr.size 是长轴/短轴直径（沿 rr.angle 的局部坐标系）
				cv::Point center((int)std::lround(rr.center.x - bbox.x),
					(int)std::lround(rr.center.y - bbox.y));

				cv::Size axes((int)std::lround(0.5f * rr.size.width * axisScale),
					(int)std::lround(0.5f * rr.size.height * axisScale));

				if (axes.width <= 0 || axes.height <= 0) return;

				cv::ellipse(mask,
					center,
					axes,
					rr.angle,        // 旋转角度（度）
					0.0, 360.0,
					255,
					-1,              // 填充
					cv::LINE_AA);
		};

		// 1.0 表示“内接于 RotatedRect”
		// 如果你想让椭圆覆盖 RotatedRect 的角（更大一圈），可以用 sqrt(2) ≈ 1.414
		fillEllipseFromRotRect(outer, outerMask, 1.0f);
		fillEllipseFromRotRect(roi, innerMask, 1.0f);

		cv::Mat1b bandMask;
		cv::bitwise_and(outerMask, ~innerMask, bandMask);

		// 采样 band 的 3D 点
		std::vector<cv::Vec3f> pts3d;
		pts3d.reserve((size_t)bbox.area() / 3);

		for (int y = 0; y < bbox.height; ++y) {
			const int v = bbox.y + y;
			const uchar* bm = bandMask.ptr<uchar>(y);
			for (int x = 0; x < bbox.width; ++x) {
				if (!bm[x]) continue;
				const int u = bbox.x + x;
				float z = depthMm(v, u);
				if (!isValidDepth(z)) continue;
				vector<cv::Point2d> uv = { cv::Point2f(u, v) }, xnyn;
				cv::undistortPoints(uv, xnyn, K, D);
				/*float xn = (u - K.at<float>(0, 2)) / K.at<float>(0, 0);
				float yn = (v - K.at<float>(1, 2)) / K.at<float>(1, 1);*/
				// 相机坐标系点（单位：mm）
				pts3d.emplace_back(xnyn[0].x * z, xnyn[0].y * z, z);
			}
		}

		if ((int)pts3d.size() < minBandSamples) return false;

		// 鲁棒平面
		cv::Vec3f n; float d;
		if (!fitPlaneRobustMAD(pts3d, n, d)) return false;

		// ROI 内回填：Z = -d / (n · dir)，dir = [xn, yn, 1]
		for (int y = 0; y < bbox.height; ++y) {
			const int v = bbox.y + y;
			const uchar* im = innerMask.ptr<uchar>(y);
			for (int x = 0; x < bbox.width; ++x) {
				if (!im[x]) continue;
				const int u = bbox.x + x;
				vector<cv::Point2d> uv = { cv::Point2f(u, v) }, xnyn;
				cv::undistortPoints(uv, xnyn, K, D);
				/*float xn = (u - K.at<float>(0, 2)) / K.at<float>(0, 0);
				float yn = (v - K.at<float>(1, 2)) / K.at<float>(1, 1);*/
				float denom = n[0] * xnyn[0].x + n[1] * xnyn[0].y + n[2] * 1.0f;
				if (std::fabs(denom) < 1e-6f) continue;

				float zHat = -d / denom;
				if (!isValidDepth(zHat)) continue;

				depthMm(v, u) = zHat;
			}
		}

		return true;
	}

	// 拟合 Z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
	// coeff = [a b c d e f]
	inline bool fitQuadLS(const std::vector<cv::Vec3f>& xynz,
		const std::vector<uint8_t>* inlierMask,
		cv::Vec<float, 6>& coeff)
	{
		// 正规方程：H(6x6), g(6x1)
		cv::Matx<double, 6, 6> H; H = cv::Matx<double, 6, 6>::zeros();
		cv::Vec<double, 6> g(0, 0, 0, 0, 0, 0);

		int nUsed = 0;
		for (size_t i = 0; i < xynz.size(); ++i) {
			if (inlierMask && (*inlierMask)[i] == 0) continue;
			const float x = xynz[i][0], y = xynz[i][1], z = xynz[i][2];

			double r0 = (double)x * x;
			double r1 = (double)y * y;
			double r2 = (double)x * y;
			double r3 = (double)x;
			double r4 = (double)y;
			double r5 = 1.0;

			double R[6] = { r0,r1,r2,r3,r4,r5 };

			for (int a = 0; a < 6; ++a) {
				g[a] += R[a] * (double)z;
				for (int b = 0; b < 6; ++b) H(a, b) += R[a] * R[b];
			}
			++nUsed;
		}

		if (nUsed < 30) return false;

		cv::Mat Hm(6, 6, CV_64F, (void*)H.val);
		cv::Mat gm(6, 1, CV_64F, (void*)g.val);
		cv::Mat cm;

		// SVD 更稳
		if (!cv::solve(Hm, gm, cm, cv::DECOMP_SVD)) return false;

		for (int i = 0; i < 6; ++i) coeff[i] = (float)cm.at<double>(i, 0);
		return true;
	}

	inline float evalQuad(const cv::Vec<float, 6>& c, float x, float y) {
		return c[0] * x * x + c[1] * y * y + c[2] * x * y + c[3] * x + c[4] * y + c[5];
	}

	static bool fitQuadRobustMAD(const std::vector<cv::Vec3f>& xynz,
		cv::Vec<float, 6>& coeff,
		float kMAD = 4.0f)
	{
		if (!fitQuadLS(xynz, nullptr, coeff)) return false;

		std::vector<float> absr; absr.reserve(xynz.size());
		std::vector<float> r(xynz.size());

		for (size_t i = 0; i < xynz.size(); ++i) {
			float x = xynz[i][0], y = xynz[i][1], z = xynz[i][2];
			float zh = evalQuad(coeff, x, y);
			float ri = z - zh;
			r[i] = ri;
			absr.push_back(std::fabs(ri));
		}

		float mad = medianOf(absr);
		if (mad < 1e-6f) mad = 1e-6f;
		float th = kMAD * mad;

		std::vector<uint8_t> inl(xynz.size(), 0);
		int cnt = 0;
		for (size_t i = 0; i < xynz.size(); ++i) {
			if (std::fabs(r[i]) <= th) { inl[i] = 1; ++cnt; }
		}
		if (cnt < 30) return true; // inlier 太少就用初值

		return fitQuadLS(xynz, &inl, coeff);
	}

	inline bool correctOneMarker_QuadEllipse(cv::Mat& depth,
		const cv::RotatedRect& roi,
		const cv::Mat& K,
		const cv::Mat& D,
		int marginPx = 12,
		int minBandSamples = 300,
		int fadePx = 0,          // 0=硬替换；>0=软过渡
		float axisScale = 1.0f)  // 1.0 椭圆内接 roi；1.414 覆盖到角
	{
		cv::Mat1f depthMm = depth;
		const int W = depthMm.cols, H = depthMm.rows;

		cv::RotatedRect outer = roi;
		outer.size.width += 2.0f * marginPx;
		outer.size.height += 2.0f * marginPx;

		cv::Rect bbox = outer.boundingRect() & cv::Rect(0, 0, W, H);
		if (bbox.width < 3 || bbox.height < 3) return false;

		cv::Mat1b innerMask(bbox.height, bbox.width, uchar(0));
		cv::Mat1b outerMask(bbox.height, bbox.width, uchar(0));

		auto fillEllipseFromRotRect = [&](const cv::RotatedRect& rr, cv::Mat1b& mask, float sc) {
			cv::Point center((int)std::lround(rr.center.x - bbox.x),
				(int)std::lround(rr.center.y - bbox.y));
			cv::Size axes((int)std::lround(0.5f * rr.size.width * sc),
				(int)std::lround(0.5f * rr.size.height * sc));
			if (axes.width <= 0 || axes.height <= 0) return;
			cv::ellipse(mask, center, axes, rr.angle, 0.0, 360.0, 255, -1, cv::LINE_AA);
		};

		fillEllipseFromRotRect(outer, outerMask, axisScale);
		fillEllipseFromRotRect(roi, innerMask, axisScale);

		cv::Mat1b bandMask;
		cv::bitwise_and(outerMask, ~innerMask, bandMask);

		// 收集 band 样本： (xn, yn, z)
		std::vector<cv::Vec3f> xynz;
		xynz.reserve((size_t)bbox.area() / 3);

		// 可选：限制最大样本数，提升实时性
		const int maxSamples = 3000;
		int step = 1;
		if (bbox.area() > 80000) step = 2; // 简单下采样策略（可按 ROI 大小调）

		for (int y = 0; y < bbox.height; y += step) {
			int v = bbox.y + y;
			const uchar* bm = bandMask.ptr<uchar>(y);
			for (int x = 0; x < bbox.width; x += step) {
				if (!bm[x]) continue;
				int u = bbox.x + x;
				float z = depthMm(v, u);
				if (!isValidDepth(z)) continue;
				vector<cv::Point2d> uv = { cv::Point2f(u, v) }, xnyn;
				cv::undistortPoints(uv, xnyn, K, D);
				/*float xn = (u - K.cx) / K.fx;
				float yn = (v - K.cy) / K.fy;*/
				xynz.emplace_back(xnyn[0].x, xnyn[0].y, z);

				if ((int)xynz.size() >= maxSamples) break;
			}
			if ((int)xynz.size() >= maxSamples) break;
		}

		if ((int)xynz.size() < minBandSamples) return false;

		// 拟合二次曲面
		cv::Vec<float, 6> c;
		if (!fitQuadRobustMAD(xynz, c, 4.0f)) return false;

		// 可选软过渡：需要 distTransform（只对 inner）
		cv::Mat1f dist;
		if (fadePx > 0) {
			cv::Mat1b inner01 = innerMask > 0;
			cv::distanceTransform(inner01, dist, cv::DIST_L2, 3);
		}

		// 回填 inner 椭圆内
		for (int y = 0; y < bbox.height; ++y) {
			int v = bbox.y + y;
			const uchar* im = innerMask.ptr<uchar>(y);
			for (int x = 0; x < bbox.width; ++x) {
				if (!im[x]) continue;
				int u = bbox.x + x;
				vector<cv::Point2d> uv = { cv::Point2f(u, v) }, xnyn;
				cv::undistortPoints(uv, xnyn, K, D);
				/*float xn = (u - K.cx) / K.fx;
				float yn = (v - K.cy) / K.fy;*/
				float zHat = evalQuad(c, xnyn[0].x, xnyn[0].y);
				if (!isValidDepth(zHat)) continue;

				if (fadePx <= 0) {
					depthMm(v, u) = zHat; // 硬替换
				}
				else {
					float a = std::min(dist(y, x) / (float)fadePx, 1.0f); // 0..1
					float zOld = depthMm(v, u);
					if (!isValidDepth(zOld)) zOld = zHat;
					depthMm(v, u) = a * zHat + (1.0f - a) * zOld; // 软过渡
				}
			}
		}

		return true;
	}
}
