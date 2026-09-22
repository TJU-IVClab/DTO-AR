#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
using namespace std;

__global__ void computeErrorAndJacobian_dense(
	const float* grad1, const float* gradxGrad1, const float* gradyGrad1,
	const float* depth1, const float3* norm1, const float* grad2, const float* depth2,
	int width, int height, double fx, double fy, double cx, double cy,
	const float* R, const float* t, const float* R_inv, const float* t_inv,
	float* JTJ, float* JTr, float* d_eMap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	int idx = y * width + x;

	// ---- ---- ---- ---- ---- geometric term ---- ---- ---- ---- ---- //
	bool is_geo_ok = true;
	float w_geo = 1;

	float Z1 = depth1[idx];
	if (Z1 <= 0) is_geo_ok = false;
	// 重建源帧中像素点的3D坐标 p1
	float X1 = (x - cx) * Z1 / fx;
	float Y1 = (y - cy) * Z1 / fy;
	float3 p1 = make_float3(X1, Y1, Z1);

	// 将点变换到目标帧坐标系 p2 = T_inv * P1 = R_inv * p1 + t_inv
	float3 p2;
	p2.x = R_inv[0] * p1.x + R_inv[1] * p1.y + R_inv[2] * p1.z + t_inv[0];
	p2.y = R_inv[3] * p1.x + R_inv[4] * p1.y + R_inv[5] * p1.z + t_inv[1];
	p2.z = R_inv[6] * p1.x + R_inv[7] * p1.y + R_inv[8] * p1.z + t_inv[2];

	float u2 = (p2.x * fx) / p2.z + cx;
	float v2 = (p2.y * fy) / p2.z + cy;
	int u2_d = __float2int_rd(u2); int v2_d = __float2int_rd(v2); // => floor()
	int u2_u = __float2int_ru(u2); int v2_u = __float2int_ru(v2); // => ceil()
	if (u2_d < 0 || u2_u >= width || v2_d < 0 || v2_u >= height) is_geo_ok = false;
	int idx2_dd = v2_d * width + u2_d; float Z2_dd = depth2[idx2_dd];
	int idx2_du = v2_u * width + u2_d; float Z2_du = depth2[idx2_du];
	int idx2_ud = v2_d * width + u2_u; float Z2_ud = depth2[idx2_ud];
	if (Z2_dd == 0 || Z2_du == 0 || Z2_ud == 0) is_geo_ok = false;
	float Z2 = Z2_dd + (u2 - u2_d) * (Z2_ud - Z2_dd) + (v2 - v2_d) * (Z2_du - Z2_dd);
	if (abs(Z2 - p2.z) > 3.0f) is_geo_ok = false; // occlusion between objects

	// 重建目标帧对应点 q 和法线 n
	float3 q2;
	q2.x = (u2 - cx) * Z2 / fx;
	q2.y = (v2 - cy) * Z2 / fy;
	q2.z = Z2;
	float3 q1;
	q1.x = R[0] * q2.x + R[1] * q2.y + R[2] * q2.z + t[0];
	q1.y = R[3] * q2.x + R[4] * q2.y + R[5] * q2.z + t[1];
	q1.z = R[6] * q2.x + R[7] * q2.y + R[8] * q2.z + t[2];

	float3 n = norm1[idx];

	// 计算误差 e = n^T * (p1 - q1), E = e^2
	float ex = p1.x - q1.x;
	float ey = p1.y - q1.y;
	float ez = p1.z - q1.z;
	float e_geo = (n.x * ex + n.y * ey + n.z * ez);
	//if (abs(e) > 3.0f) return;

	// 计算误差对位姿参数的Jacobian（6维：平移vx,vy,vz; 旋转ωx,ωy,ωz）
	// 平移部分: ∂e/∂v = -n^T
	float Jt_geo[6];
	Jt_geo[0] = -n.x;  // ∂e/∂vx
	Jt_geo[1] = -n.y;  // ∂e/∂vy
	Jt_geo[2] = -n.z;  // ∂e/∂vz
	// 旋转部分: ∂e/∂ω ≈ -n^T * -[q1]_x (叉积矩阵)
	// 等价于 Jt[3..5] = (n × q1)
	float px = q1.x, py = q1.y, pz = q1.z;
	Jt_geo[3] = (n.y * pz - n.z * py); // ∂e/∂ωx
	Jt_geo[4] = (n.z * px - n.x * pz); // ∂e/∂ωy
	Jt_geo[5] = (n.x * py - n.y * px); // ∂e/∂ωz


	// ---- ---- ---- ---- ---- photometric term ---- ---- ---- ---- ---- //
	bool is_pho_ok = true;
	float alpha_pho = 25.0f;
	float w_pho = 2e-2 / 3e2;

	if (x < 20 || x >= width - 20 || y < 20 || y >= height - 20) is_pho_ok = false;
	float I2 = grad2[idx];
	if (I2 >= 2e3) is_pho_ok = false;
	float Z02 = depth2[idx];
	if (Z02 <= 0) is_pho_ok = false;

	float X2 = (x - cx) * Z02 / fx;
	float Y2 = (y - cy) * Z02 / fy;
	float3 p02 = make_float3(X2, Y2, Z02);

	float3 p01;
	p01.x = R[0] * p02.x + R[1] * p02.y + R[2] * p02.z + t[0];
	p01.y = R[3] * p02.x + R[4] * p02.y + R[5] * p02.z + t[1];
	p01.z = R[6] * p02.x + R[7] * p02.y + R[8] * p02.z + t[2];

	float u1 = (p01.x * fx) / p01.z + cx;
	float v1 = (p01.y * fy) / p01.z + cy;
	int u1_d = __float2int_rd(u1); int v1_d = __float2int_rd(v1); // => floor()
	int u1_u = __float2int_ru(u1); int v1_u = __float2int_ru(v1); // => ceil()
	int u1_n = __float2int_rn(u1); int v1_n = __float2int_rn(v1);
	if (u1_d < 20 || u1_u >= width - 20 || v1_d < 20 || v1_u >= height - 20) is_pho_ok = false;
	int idx1_dd = v1_d * width + u1_d; float I1_dd = grad1[idx1_dd];
	int idx1_du = v1_u * width + u1_d; float I1_du = grad1[idx1_du];
	int idx1_ud = v1_d * width + u1_u; float I1_ud = grad1[idx1_ud];
	int idx1_nn = v1_n * width + u1_n;
	float I1 = I1_dd + (u1 - u1_d) * (I1_ud - I1_dd) + (v1 - v1_d) * (I1_du - I1_dd);

	float e_pho = (I2 - I1) * alpha_pho;
	//if (abs(e) > 2e2f) return;

	// 计算误差对位姿参数的Jacobian（6维：平移vx,vy,vz; 旋转ωx,ωy,ωz）
	float ggx = gradxGrad1[idx1_nn];
	float ggy = gradyGrad1[idx1_nn];
	float Jt_pho[6];
	// 平移部分: ∂e/∂v = -∇I1(u1) * ∂π/∂p1 * E
	Jt_pho[0] = -ggx * (fx / p01.z);  // ∂e/∂vx
	Jt_pho[1] = -ggy * (fy / p01.z);  // ∂e/∂vy
	Jt_pho[2] = -(Jt_pho[0] * p01.x + Jt_pho[1] * p01.y) / p01.z;  // ∂e/∂vz
	// 旋转部分: ∂e/∂ω ≈ -∇I1(u1) * ∂π/∂p1 * -[p1]_x (叉积矩阵)
	Jt_pho[3] = p01.y * Jt_pho[2] + fy * ggy; // ∂e/∂ωx
	Jt_pho[4] = -p01.x * Jt_pho[2] - fx * ggx; // ∂e/∂ωy
	Jt_pho[5] = -Jt_pho[0] * p01.y + Jt_pho[1] * p01.x; // ∂e/∂ωz


	// ---- ---- ---- ---- ---- add up ---- ---- ---- ---- ---- //
	if (is_geo_ok) {
		d_eMap[idx] = d_eMap[idx] + e_geo;
		// 原子累加到全局JTJ和JTr
		for (int i = 0; i < 6; ++i) {
			atomicAdd(&JTr[i], -w_geo * e_geo * Jt_geo[i]);  // J^T * r (这里累加 -e*Jt)
			for (int j = 0; j <= i; ++j) {
				atomicAdd(&JTJ[i * 6 + j], w_geo * Jt_geo[i] * Jt_geo[j]);
				if (i != j) {
					atomicAdd(&JTJ[j * 6 + i], w_geo * Jt_geo[i] * Jt_geo[j]);
				}
			}
		}
	}
	if (is_pho_ok) {
		d_eMap[idx] = d_eMap[idx] + e_pho / 2e3;
		// 原子累加到全局JTJ和JTr
		for (int i = 0; i < 6; ++i) {
			atomicAdd(&JTr[i], -w_pho * e_pho * Jt_pho[i]);  // J^T * r (这里累加 -e*Jt)
			for (int j = 0; j <= i; ++j) {
				atomicAdd(&JTJ[i * 6 + j], w_pho * Jt_pho[i] * Jt_pho[j]);
				if (i != j) {
					atomicAdd(&JTJ[j * 6 + i], w_pho * Jt_pho[i] * Jt_pho[j]);
				}
			}
		}
	}
}

// 优化函数：输入两帧的深度/法线图和初始相对位姿，返回优化后的相对位姿
Eigen::Matrix4f denseOptPoseSE3(
	const cv::Mat& grad1, const cv::Mat& gradxGrad1, const cv::Mat& gradyGrad1, const cv::Mat& depth1, const cv::Mat& norm1, const cv::Mat& grad2, const cv::Mat& depth2,
	const Eigen::Matrix4f& initPose, int maxIter = 20, float tol = 1e-6)
{
	// 摄像机内参（示例值，应根据实际情况设置）
	double fx = 1154.93, fy = 1154.95;
	double cx = 560.74, cy = 400.553;

	int width = depth1.cols, height = depth1.rows;
	size_t imgSize = width * height;

	// 分配GPU内存并复制深度和法线数据
	float* d_grad1; float* d_gradxGrad1; float* d_gradyGrad1; float* d_depth1; float3* d_norm1;
	float* d_grad2; float* d_depth2;
	cudaMalloc(&d_grad1, imgSize * sizeof(float)); cudaMemcpy(d_grad1, grad1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_gradxGrad1, imgSize * sizeof(float)); cudaMemcpy(d_gradxGrad1, gradxGrad1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_gradyGrad1, imgSize * sizeof(float)); cudaMemcpy(d_gradyGrad1, gradyGrad1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_depth1, imgSize * sizeof(float)); cudaMemcpy(d_depth1, depth1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_norm1, imgSize * sizeof(float3)); cudaMemcpy(d_norm1, norm1.ptr<float>(), imgSize * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMalloc(&d_grad2, imgSize * sizeof(float)); cudaMemcpy(d_grad2, grad2.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_depth2, imgSize * sizeof(float)); cudaMemcpy(d_depth2, depth2.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);

	// debug: check error map
	float* d_eMap;
	cudaMalloc(&d_eMap, imgSize * sizeof(float));

	// 为JTJ和JTr分配GPU内存
	float* d_JTr;  /*梯度向量6dof*/ float* d_JTJ;  /* Hessian 矩阵 (6x6 = 36)*/
	cudaMalloc(&d_JTr, 6 * sizeof(float));
	cudaMalloc(&d_JTJ, 36 * sizeof(float));

	// CUDA线程块与网格配置
	dim3 threads(16, 16);
	dim3 blocks((width + 15) / 16, (height + 15) / 16);

	// 将初始位姿拆分为旋转和平移
	Eigen::Matrix4f initPoseinv = initPose.inverse();
	Eigen::Matrix3f R = initPose.block<3, 3>(0, 0), R_inv = initPoseinv.block<3, 3>(0, 0);
	Eigen::Vector3f t = initPose.block<3, 1>(0, 3), t_inv = initPoseinv.block<3, 1>(0, 3);

	for (int iter = 0; iter < maxIter; ++iter) {
		// 清零JTJ和JTr
		cudaMemset(d_JTr, 0, 6 * sizeof(float));
		cudaMemset(d_JTJ, 0, 36 * sizeof(float));
		cudaMemset(d_eMap, 0, imgSize * sizeof(float)); // debug

		// 准备当前旋转矩阵和平移向量数据
		float h_R[9] = { R(0,0), R(0,1), R(0,2),
						 R(1,0), R(1,1), R(1,2),
						 R(2,0), R(2,1), R(2,2) },
			h_R_inv[9] = { R_inv(0,0), R_inv(0,1), R_inv(0,2),
						   R_inv(1,0), R_inv(1,1), R_inv(1,2),
						   R_inv(2,0), R_inv(2,1), R_inv(2,2) };
		float h_t[3] = { t(0), t(1), t(2) },
			h_t_inv[3] = { t_inv(0), t_inv(1), t_inv(2) };

		// 将旋转和平移拷贝到GPU常量缓冲区（或全局内存）
		float* d_R; float* d_t;
		float* d_R_inv; float* d_t_inv;
		cudaMalloc(&d_R, 9 * sizeof(float)); cudaMemcpy(d_R, h_R, 9 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc(&d_t, 3 * sizeof(float)); cudaMemcpy(d_t, h_t, 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc(&d_R_inv, 9 * sizeof(float)); cudaMemcpy(d_R_inv, h_R_inv, 9 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc(&d_t_inv, 3 * sizeof(float)); cudaMemcpy(d_t_inv, h_t_inv, 3 * sizeof(float), cudaMemcpyHostToDevice);

		// 调用CUDA核函数计算误差和Jacobian累积
		computeErrorAndJacobian_dense << <blocks, threads >> > (
			d_grad1, d_gradxGrad1, d_gradyGrad1, d_depth1, d_norm1, d_grad2, d_depth2,
			width, height, fx, fy, cx, cy,
			d_R, d_t, d_R_inv, d_t_inv,
			d_JTJ, d_JTr,
			d_eMap
			);
		cudaDeviceSynchronize();

		// 释放临时常量缓冲区
		cudaFree(d_R);
		cudaFree(d_t);
		cudaFree(d_R_inv);
		cudaFree(d_t_inv);

		// 将累积结果从GPU拷贝回CPU
		float h_JTr[6], h_JTJ[36]; std::vector<float> h_eMap(width * height);
		cudaMemcpy(h_JTr, d_JTr, 6 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_JTJ, d_JTJ, 36 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_eMap.data(), d_eMap, imgSize * sizeof(float), cudaMemcpyDeviceToHost); // debug

		// 构造Eigen矩阵并求解增量 Δξ = (J^T J)^{-1} (-J^T r)
		Eigen::Matrix<float, 6, 6> H;
		Eigen::Matrix<float, 6, 1> g;
		for (int i = 0; i < 6; ++i) {
			g(i) = h_JTr[i];
			for (int j = 0; j < 6; ++j) {
				H(i, j) = h_JTJ[i * 6 + j];
			}
		}
		Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(g);

		// debug
		cout << "Iter[" << iter << "] delta: " << delta.norm() << endl;
		Eigen::Matrix4f Tij = Eigen::Matrix4f::Identity();
		Tij.block<3, 3>(0, 0) = R;
		Tij.block<3, 1>(0, 3) = t;
		cout << "Iter[" << iter << "] Tij: " << endl << Tij << endl << endl;
		cv::Mat errorMat(height, width, CV_32FC1, h_eMap.data());
		cv::imwrite("error_map.tif", errorMat);
		cv::Mat absMat, validMat, display;
		absMat = abs(errorMat);
		validMat = absMat > 0;
		validMat /= 255;
		cout << "Average pixel error: " << cv::sum(absMat)[0] / cv::sum(validMat)[0] << "mm" << endl;
		cout << "Valid pixel count: " << cv::sum(validMat)[0] << " s" << endl;
		cv::normalize(absMat, absMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::applyColorMap(absMat, display, cv::COLORMAP_JET);
		cv::imshow("error map from cuda", display);
		cv::waitKey(0);

		// 检查收敛条件
		if (delta.norm() < tol) break;

		// 构造SE(3)增量变换：旋转部分使用Rodrigues公式
		Eigen::Vector3f omega(delta(3), delta(4), delta(5));
		Eigen::Vector3f upsilon(delta(0), delta(1), delta(2));
		// Rodrigues: 旋转向量 -> 3x3 矩阵
		cv::Mat rvec = (cv::Mat_<float>(3, 1) << omega(0), omega(1), omega(2));
		cv::Mat Rinc_cv;
		cv::Rodrigues(rvec, Rinc_cv);
		Eigen::Matrix3f Rinc;
		Rinc << Rinc_cv.at<float>(0, 0), Rinc_cv.at<float>(0, 1), Rinc_cv.at<float>(0, 2),
			Rinc_cv.at<float>(1, 0), Rinc_cv.at<float>(1, 1), Rinc_cv.at<float>(1, 2),
			Rinc_cv.at<float>(2, 0), Rinc_cv.at<float>(2, 1), Rinc_cv.at<float>(2, 2);
		// 更新当前位姿：左乘增量变换
		R = Rinc * R;
		t = Rinc * t + upsilon;
		Eigen::Matrix4f Tnew = Eigen::Matrix4f::Identity(), Tnewinv;
		Tnew.block<3, 3>(0, 0) = R;
		Tnew.block<3, 1>(0, 3) = t;
		Tnewinv = Tnew.inverse();
		R_inv = Tnewinv.block<3, 3>(0, 0);
		t_inv = Tnewinv.block<3, 1>(0, 3);
	}

	// 释放GPU内存
	cudaFree(d_grad1); cudaFree(d_gradxGrad1); cudaFree(d_gradyGrad1); cudaFree(d_depth1); cudaFree(d_norm1);
	cudaFree(d_grad2); cudaFree(d_depth2);
	cudaFree(d_JTr); cudaFree(d_JTJ);

	// 返回优化后的位姿矩阵
	Eigen::Matrix4f Topt = Eigen::Matrix4f::Identity();
	Topt.block<3, 3>(0, 0) = R;
	Topt.block<3, 1>(0, 3) = t;
	return Topt;
}