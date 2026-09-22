#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include <memory>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/linear_solver_dense.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/types_six_dof_expmap.h> // VertexSE3Expmap

#ifndef ORB_SLAM3_SPOTOPTIMIZATION_H
#define ORB_SLAM3_SPOTOPTIMIZATION_H

namespace ORB_SLAM3 {
	// --------------------- @measurement ---------------------
	struct MeasBoundary {
		std::shared_ptr<std::vector<Eigen::Vector2d>> uk;   // 最近邻 edgel 的亚像素位置  (tilde u)
		std::shared_ptr<std::vector<Eigen::Vector2d>> nk;    // edgel 的像面单位法向     (2D)
		double sqrt_w = 1.0;  // 权重的平方根
		Eigen::Vector3d vk;   // 该边对应的圆上采样向量 v_k（在圆点局部坐标系）
	};
	struct MeasDepth {
		Eigen::Vector3d x_cam; // 相机系测量 3D \tilde{x}
		double sqrt_w = 1.0;
	};
	struct MeasNormal {
		Eigen::Vector3d n_cam; // 相机系单位法向测量
		double sqrt_w = 1.0;
	};
	struct CameraIntrinsics {
		double fx = 0, fy = 0, cx = 0, cy = 0;
	};

	// --------------------- tool functions ---------------------
	//static Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
	//	Eigen::Matrix3d S;
	//	S << 0, -v.z(), v.y(),
	//		v.z(), 0, -v.x(),
	//		-v.y(), v.x(), 0;
	//	return S;
	//}
	// 从两向量构造四元数（把 ez 旋到 n）
	inline Eigen::Quaterniond quatFromTwoVectors(
		const Eigen::Vector3d& a, const Eigen::Vector3d& b)
	{
		Eigen::Vector3d v = a.normalized().cross(b.normalized());
		double w = std::sqrt((a.squaredNorm()) * (b.squaredNorm())) + a.normalized().dot(b.normalized());
		Eigen::Quaterniond q(w, v.x(), v.y(), v.z());
		q.normalize();
		//std::cout << b.matrix() << std::endl << q.toRotationMatrix().matrix() << std::endl;
		return q;
	}
	// 投影及其雅可比 Jproj(x)
	inline void projectWithJacobian(
		const Eigen::Vector3d& x,
		const CameraIntrinsics* K,
		Eigen::Vector2d& u,
		Eigen::Matrix<double, 2, 3>& Jproj)
	{
		const double X = x.x(), Y = x.y(), Z = x.z();
		const double invZ = 1.0 / Z;
		u.x() = K->fx * X * invZ + K->cx;
		u.y() = K->fy * Y * invZ + K->cy;

		Jproj.setZero();
		Jproj(0, 0) = K->fx * invZ;
		Jproj(0, 2) = -K->fx * X * invZ * invZ;
		Jproj(1, 1) = K->fy * invZ;
		Jproj(1, 2) = -K->fy * Y * invZ * invZ;
	}
	inline void getNearest_U_N(
		const Eigen::Vector2d& u,
		const std::shared_ptr<std::vector<Eigen::Vector2d>>& uk,
		const std::shared_ptr<std::vector<Eigen::Vector2d>>& nk,
		Eigen::Vector2d& ua,
		Eigen::Vector2d& na)
	{
		assert(uk && nk);
		assert(uk->size() == nk->size());
		assert(!uk->empty());

		double bestDist2 = std::numeric_limits<double>::infinity();
		std::size_t bestIdx = 0;

		const auto& Uk = *uk;
		const auto& Nk = *nk;

		for (std::size_t i = 0; i < Uk.size(); ++i) {
			Eigen::Vector2d diff = Uk[i] - u;
			double dist2 = diff.squaredNorm();
			if (dist2 < bestDist2) {
				bestDist2 = dist2;
				bestIdx = i;

				// 完全相同的点，可以直接提前返回
				if (bestDist2 == 0.0) {
					break;
				}
			}
		}
		ua = Uk[bestIdx];
		na = Nk[bestIdx];
		//std::cout << "ua - u: " << (ua - u).x() << ", " << (ua - u).y() << std::endl;
	}
	// --------------------- vertex: spot ---------------------
	struct DotEstimate {
		Eigen::Vector3d c;            // 圆心（世界系）
		Eigen::Quaterniond q;         // 朝向（z 轴为法向）
		DotEstimate() : c(Eigen::Vector3d::Zero()), q(Eigen::Quaterniond::Identity()) {}
	};

	class VertexDot final : public g2o::BaseVertex<6, DotEstimate> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			void setToOriginImpl() override {
			_estimate = DotEstimate();
		}
		void oplusImpl(const double* update_) override {
			Eigen::Map<const Eigen::Matrix<double, 6, 1>> upd(update_);
			_estimate.c += upd.head<3>();
			Eigen::Vector3d dphi = upd.tail<3>();
			double theta = dphi.norm();
			Eigen::Quaterniond dq = (theta < 1e-12)
				? Eigen::Quaterniond(1, 0.5 * dphi.x(), 0.5 * dphi.y(), 0.5 * dphi.z())
				: Eigen::Quaterniond(Eigen::AngleAxisd(theta, dphi.normalized()));
			_estimate.q = (dq * _estimate.q).normalized();
		}
		bool read(std::istream&) override { return false; }
		bool write(std::ostream&) const override { return false; }

		inline const Eigen::Vector3d& c()  const { return _estimate.c; }
		inline Eigen::Matrix3d R() const { return _estimate.q.toRotationMatrix(); }
	};

	// --------------------- edge：boundary factor（1 dof） ---------------------
	class EdgeDotBoundary final
		: public g2o::BaseBinaryEdge<1, MeasBoundary, VertexDot, g2o::VertexSE3Expmap> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			explicit EdgeDotBoundary(const CameraIntrinsics* K) : K_(*K) {}

		void setMeasurement(const MeasBoundary& measurement) override {
			_measurement = measurement;
			nearestValid_ = false;
		}

		void computeError() override {
			const auto* vDot = static_cast<const VertexDot*>(_vertices[0]);
			const auto* vCam = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

			// 世界点：p = c + R * v_k
			Eigen::Vector3d p = vDot->c() + vDot->R() * _measurement.vk;

			// 相机系：x = Rcw * p + t
			Eigen::Vector3d x = vCam->estimate().map(p); // map() = R*p + t
			Eigen::Vector2d u; Eigen::Matrix<double, 2, 3> Jproj;
			projectWithJacobian(x, &K_, u, Jproj);

			Eigen::Vector2d ua, na;
			nearest(u, ua, na);
			const Eigen::Vector2d du = u - ua;
			const double r = _measurement.sqrt_w * na.dot(du);
			_error[0] = r;
			//std::cout << "du: " << du << " na: " << na << " r: " << r << std::endl;
		}

		void linearizeOplus() override {
			const auto* vDot = static_cast<const VertexDot*>(_vertices[0]);
			const auto* vCam = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

			const Eigen::Matrix3d Rcw = vCam->estimate().rotation().toRotationMatrix();
			const Eigen::Vector3d t = vCam->estimate().translation();
			const Eigen::Matrix3d Rdi = vDot->R();

			const Eigen::Vector3d p = vDot->c() + Rdi * _measurement.vk;
			const Eigen::Vector3d x = Rcw * p + t;

			Eigen::Vector2d u; Eigen::Matrix<double, 2, 3> Jproj;
			projectWithJacobian(x, &K_, u, Jproj);

			Eigen::Vector2d ua, na;
			nearest(u, ua, na);
			const Eigen::RowVector2d nT = _measurement.sqrt_w * na.transpose();

			_jacobianOplusXi.setZero();
			if (!vDot->fixed()) {
				// du/dc = Jproj * Rcw
				Eigen::Matrix<double, 1, 3> J_c = nT * Jproj * Rcw;

				// du/dphi = Jproj * Rcw * ( - [ Rdi * vk ]_x )
				Eigen::Matrix<double, 1, 3> J_phi =
					nT * Jproj * Rcw * (-Rdi * Skew(_measurement.vk) * Rdi.transpose());

				_jacobianOplusXi.block<1, 3>(0, 0) = J_c;
				_jacobianOplusXi.block<1, 3>(0, 3) = J_phi;
			}

			_jacobianOplusXj.setZero();
			if (!vCam->fixed()) {
				// du/dxi = Jproj * [ -[x]_x  I ]
				Eigen::Matrix<double, 2, 6> J_xi;
				J_xi.setZero();
				J_xi.block<2, 3>(0, 0) = Jproj * (-Skew(x));
				J_xi.block<2, 3>(0, 3) = Jproj;

				Eigen::Matrix<double, 1, 6> J_cam = nT * J_xi;

				_jacobianOplusXj = J_cam;    // wrt VertexSE3Expmap (6)
			}
		}

		bool read(std::istream&) override { return false; }
		bool write(std::ostream&) const override { return false; }

		const CameraIntrinsics K_;

	private:
		// Builders finish uk/nk before creating edges; observations stay immutable.
		// Exact finite queries only: preserve scan order and nearest-point ties.
		void nearest(const Eigen::Vector2d& u, Eigen::Vector2d& ua, Eigen::Vector2d& na) {
			if (!nearestValid_ || !u.allFinite() ||
				u.x() != nearestU_.x() || u.y() != nearestU_.y()) {
				getNearest_U_N(u, _measurement.uk, _measurement.nk, nearestUa_, nearestNa_);
				nearestU_ = u;
				nearestValid_ = u.allFinite();
			}
			ua = nearestUa_;
			na = nearestNa_;
		}
		bool nearestValid_ = false;
		Eigen::Vector2d nearestU_, nearestUa_, nearestNa_;
	};

	// --------------------- edge：depth factor（3 dof） ---------------------
	class EdgeStereoSE3ProjectDot
		: public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexDot, g2o::VertexSE3Expmap> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

			EdgeStereoSE3ProjectDot() {}

		bool read(std::istream& is) override { return false; }
		bool write(std::ostream& os) const override { return false; }

		// 残差：obs - 投影
		void computeError() override {
			const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
			const VertexDot* v2 = static_cast<const VertexDot*>(_vertices[0]);

			Eigen::Vector3d obs = _measurement;
			// 世界点位置用 VertexDot 的 c()
			Eigen::Vector3d Pw = v2->c();
			Eigen::Vector3d Pc = v1->estimate().map(Pw);

			_error = obs - cam_project(Pc, bf);
		}

		bool isDepthPositive() {
			const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
			const VertexDot* v2 = static_cast<const VertexDot*>(_vertices[0]);
			Eigen::Vector3d Pw = v2->c();
			Eigen::Vector3d Pc = v1->estimate().map(Pw);
			return Pc(2) > 0.0;
		}

		void linearizeOplus() override {
			// 相机位姿顶点（和原来一样）
			g2o::VertexSE3Expmap* vj = static_cast<g2o::VertexSE3Expmap*>(_vertices[1]);
			g2o::SE3Quat T(vj->estimate());

			// 点顶点：现在是 VertexDot
			VertexDot* vi = static_cast<VertexDot*>(_vertices[0]);
			Eigen::Vector3d xyz = vi->c();               // 只用位置部分
			Eigen::Vector3d xyz_tr = T.map(xyz);

			const Eigen::Matrix3d R = T.rotation().toRotationMatrix();

			const double x = xyz_tr[0];
			const double y = xyz_tr[1];
			const double z = xyz_tr[2];
			const double z_2 = z * z;

			// --------- 对顶点 Xi（VertexDot）的雅可比：3x6 ---------
			// 先清零，后 3 列（姿态部分）保持为 0
			_jacobianOplusXi.setZero();

			// 位置 c 的部分（列 0~2），照抄你原来的 VertexSBAPointXYZ 版本
			_jacobianOplusXi(0, 0) = -fx * R(0, 0) / z + fx * x * R(2, 0) / z_2;
			_jacobianOplusXi(0, 1) = -fx * R(0, 1) / z + fx * x * R(2, 1) / z_2;
			_jacobianOplusXi(0, 2) = -fx * R(0, 2) / z + fx * x * R(2, 2) / z_2;

			_jacobianOplusXi(1, 0) = -fy * R(1, 0) / z + fy * y * R(2, 0) / z_2;
			_jacobianOplusXi(1, 1) = -fy * R(1, 1) / z + fy * y * R(2, 1) / z_2;
			_jacobianOplusXi(1, 2) = -fy * R(1, 2) / z + fy * y * R(2, 2) / z_2;

			_jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - bf * R(2, 0) / z_2;
			_jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) - bf * R(2, 1) / z_2;
			_jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2) - bf * R(2, 2) / z_2;

			// 姿态 q 的 3 个增量参数（列 3,4,5）对当前误差没有影响，保持 0 即可

			// --------- 对顶点 Xj（VertexSE3Expmap）的雅可比：3x6 ---------
			_jacobianOplusXj(0, 0) = x * y / z_2 * fx;
			_jacobianOplusXj(0, 1) = -(1.0 + (x * x / z_2)) * fx;
			_jacobianOplusXj(0, 2) = y / z * fx;
			_jacobianOplusXj(0, 3) = -1.0 / z * fx;
			_jacobianOplusXj(0, 4) = 0.0;
			_jacobianOplusXj(0, 5) = x / z_2 * fx;

			_jacobianOplusXj(1, 0) = (1.0 + y * y / z_2) * fy;
			_jacobianOplusXj(1, 1) = -x * y / z_2 * fy;
			_jacobianOplusXj(1, 2) = -x / z * fy;
			_jacobianOplusXj(1, 3) = 0.0;
			_jacobianOplusXj(1, 4) = -1.0 / z * fy;
			_jacobianOplusXj(1, 5) = y / z_2 * fy;

			_jacobianOplusXj(2, 0) = _jacobianOplusXj(0, 0) - bf * y / z_2;
			_jacobianOplusXj(2, 1) = _jacobianOplusXj(0, 1) + bf * x / z_2;
			_jacobianOplusXj(2, 2) = _jacobianOplusXj(0, 2);
			_jacobianOplusXj(2, 3) = _jacobianOplusXj(0, 3);
			_jacobianOplusXj(2, 4) = 0.0;
			_jacobianOplusXj(2, 5) = _jacobianOplusXj(0, 5) - bf / z_2;
		}

		// 和原来一样的投影函数
		Eigen::Vector3d cam_project(const Eigen::Vector3d& trans_xyz, const float& bf) const {
			const double x = trans_xyz[0];
			const double y = trans_xyz[1];
			const double z = trans_xyz[2];
			const double invz = 1.0 / z;
			const double uL = fx * x * invz + cx;
			const double v = fy * y * invz + cy;
			const double uR = uL - bf * invz;
			Eigen::Vector3d res;
			res << uL, v, uR;
			return res;
		}

		double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0, bf = 0.0;
	};
	class EdgeDotDepth final
		: public g2o::BaseBinaryEdge<3, MeasDepth, VertexDot, g2o::VertexSE3Expmap> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			void computeError() override {
			const auto* vDot = static_cast<const VertexDot*>(_vertices[0]);
			const auto* vCam = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

			const Eigen::Vector3d xhat = vCam->estimate().rotation().toRotationMatrix() * vDot->c()
				+ vCam->estimate().translation();
			_error = _measurement.sqrt_w * (_measurement.x_cam - xhat);
		}

		void linearizeOplus() override {
			const auto* vDot = static_cast<const VertexDot*>(_vertices[0]);
			const auto* vCam = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
			const Eigen::Matrix3d Rcw = vCam->estimate().rotation().toRotationMatrix();
			const Eigen::Vector3d xhat = Rcw * vDot->c() + vCam->estimate().translation();

			// wrt dot: [-sqrt(w) * Rcw | 0]
			_jacobianOplusXi.setZero();
			_jacobianOplusXi.block<3, 3>(0, 0) = -_measurement.sqrt_w * Rcw;

			// wrt pose: -sqrt(w) * [ -[x]_x  I ]
			_jacobianOplusXj.setZero();
			_jacobianOplusXj.block<3, 3>(0, 0) = -_measurement.sqrt_w * (-Skew(xhat));
			_jacobianOplusXj.block<3, 3>(0, 3) = -_measurement.sqrt_w * Eigen::Matrix3d::Identity();
		}

		bool read(std::istream&) override { return false; }
		bool write(std::ostream&) const override { return false; }
	};

	// --------------------- edge：normal factor（3 dof） ---------------------
	class EdgeDotNormal final
		: public g2o::BaseBinaryEdge<3, MeasNormal, VertexDot, g2o::VertexSE3Expmap> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			void computeError() override {
			const auto* vDot = static_cast<const VertexDot*>(_vertices[0]);
			const auto* vCam = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

			const Eigen::Matrix3d Rcw = vCam->estimate().rotation().toRotationMatrix();
			const Eigen::Vector3d ez(0, 0, 1);
			const Eigen::Vector3d n_hat = Rcw * vDot->R() * ez;
			_error = _measurement.sqrt_w * (_measurement.n_cam - n_hat);
			//std::cout << "normal factor error: " << _error.x() << " " << _error.y() << " " << _error.z() << " " << _error.norm() << std::endl;
		}

		void linearizeOplus() override {
			const auto* vDot = static_cast<const VertexDot*>(_vertices[0]);
			const auto* vCam = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

			const Eigen::Matrix3d Rcw = vCam->estimate().rotation().toRotationMatrix();
			const Eigen::Matrix3d Rdi = vDot->R();
			const Eigen::Vector3d ez(0, 0, 1);
			const Eigen::Vector3d n_hat = Rcw * Rdi * ez;

			_jacobianOplusXi.setZero();
			// dr/dphi = + sqrt(w) * Rcw * [ Rdi * ez ]_x
			//_jacobianOplusXi.block<3, 3>(0, 3) = _measurement.sqrt_w * (Rcw * Rdi * Skew(ez));
			_jacobianOplusXi.block<3, 3>(0, 3) =
				_measurement.sqrt_w * (Rcw * Rdi * Skew(ez) * Rdi.transpose());
			_jacobianOplusXj.setZero();
			// dr/domega = + sqrt(w) * [ n_hat ]_x
			_jacobianOplusXj.block<3, 3>(0, 0) = _measurement.sqrt_w * Skew(n_hat);
			// dr/dt = 0
		}

		bool read(std::istream&) override { return false; }
		bool write(std::ostream&) const override { return false; }
	};
}
#endif //ORB_SLAM3_SPOTOPTIMIZATION_H