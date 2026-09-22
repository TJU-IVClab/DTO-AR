//#include "DenseOptimizer.cuh"
//
//#include <thrust/device_ptr.h>
//#include <thrust/inner_product.h>
//
//inline __device__ float2 reproject_ideal_to_distorted(const float2& p_ideal_center, cuCam cam)
//{
//	// ====================================================
//	//  Ideal pixel coordinates -> Normalized coordinates (xn, yn) [Double calculation]
//	// ====================================================
//	double xn = (/*static_cast<double>*/(p_ideal_center.x) - cam.cx) / cam.fx;
//	double yn = (/*static_cast<double>*/(p_ideal_center.y) - cam.cy) / cam.fy;
//
//	// ====================================================
//	//  Apply the radial and tangential distortion model [Double calculation]
//	// ====================================================
//
//	// 3.2. Compute radial distance squared
//	double r_sq = xn * xn + yn * yn;
//
//	// 3.3. Compute radial distortion factor
//	double radial_factor = 1.0 + cam.k1 * r_sq + cam.k2 * r_sq * r_sq + cam.k3 * r_sq * r_sq * r_sq;
//
//	// 3.4. Compute radial and tangential distortion components
//	double x_radial = xn * radial_factor;
//	double y_radial = yn * radial_factor;
//
//	double x_tan = 2.0 * cam.p1 * xn * yn + cam.p2 * (r_sq + 2.0 * xn * xn);
//	double y_tan = cam.p1 * (r_sq + 2.0 * yn * yn) + 2.0 * cam.p2 * xn * yn;
//
//	// 3.5. Combine to get the distorted normalized coordinates (xd, yd)
//	double xd = x_radial + x_tan;
//	double yd = y_radial + y_tan;
//
//	// ====================================================
//	//  Distorted normalized coordinates -> Distorted pixel coordinates [convert back to float output]
//	// ====================================================
//	float u_distorted = /*static_cast<float>*/(xd * cam.fx + cam.cx);
//	float v_distorted = /*static_cast<float>*/(yd * cam.fy + cam.cy);
//
//	return make_float2(u_distorted, v_distorted);
//}
//inline __device__ float subsample_bilinear(const float* img, const float2& u_dist, const int width, const int height)
//{
//	// Left top integer coordinates
//	int x0 = __float2int_rd(u_dist.x);
//	int y0 = __float2int_rd(u_dist.y);
//	int x1 = x0 + 1;
//	int y1 = y0 + 1;
//	// boundary checks
//	if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height) return .0f;
//	// Get values from the image
//	float I00 = img[y0 * width + x0];
//	float I10 = img[y0 * width + x1];
//	float I01 = img[y1 * width + x0];
//	float I11 = img[y1 * width + x1];
//	if (I00 == .0f || I10 == .0f || I01 == .0f || I11 == .0f) return .0f;
//	// Compute dx and dy
//	float dx = u_dist.x - static_cast<float>(x0);
//	float dy = u_dist.y - static_cast<float>(y0);
//	// Interpolate along x (horizontal)
//	float I0 = I00 * (1.0f - dx) + I10 * dx; // Top interpolation
//	float I1 = I01 * (1.0f - dx) + I11 * dx; // Bottom interpolation
//	// Final interpolation along y (vertical)
//	return I0 * (1.0f - dy) + I1 * dy;
//}
//
//__device__ bool computeGeoResidualAndJacobianPixel(
//	int x, int y, int idx,
//	const float* depth1, const float3* norm1, const float* depth2,
//	cuCam cam, const double2* raymap, cuRelPose pose,
//	float& e_geo, float Jt_geo[6]
//) {
//	float Z1 = depth1[idx] * 1e-3;
//	if (Z1 <= 0) return false;
//
//	float3 n = norm1[idx];
//
//	float3 P1; // 重建源帧中像素点的3D坐标 p1
//	P1.x = raymap[idx].x * Z1;
//	P1.y = raymap[idx].y * Z1;
//	P1.z = Z1;
//
//	float3 P2; // 将点变换到目标帧坐标系 p2 = T_inv * P1 = R_inv * p1 + t_inv
//	P2.x = pose.R_inv[0] * P1.x + pose.R_inv[1] * P1.y + pose.R_inv[2] * P1.z + pose.t_inv[0];
//	P2.y = pose.R_inv[3] * P1.x + pose.R_inv[4] * P1.y + pose.R_inv[5] * P1.z + pose.t_inv[1];
//	P2.z = pose.R_inv[6] * P1.x + pose.R_inv[7] * P1.y + pose.R_inv[8] * P1.z + pose.t_inv[2];
//
//	float2 u_ideal, u_dist;
//	u_ideal.x = (P2.x * cam.fx) / P2.z + cam.cx;
//	u_ideal.y = (P2.y * cam.fy) / P2.z + cam.cy;
//	u_dist = reproject_ideal_to_distorted(u_ideal, cam);
//	float Z2 = subsample_bilinear(depth2, u_dist, cam.width, cam.height) * 1e-3;
//	//float Z2 = P2.z;
//	if (Z2 == .0f) return false;
//
//	if (abs(Z2 - P2.z) > 3e-3f) return false; // occlusion between objects
//
//	float3 Q2; // 重建目标帧对应点 q
//	Q2.x = (u_ideal.x - cam.cx) * Z2 / cam.fx;
//	Q2.y = (u_ideal.y - cam.cy) * Z2 / cam.fy;
//	Q2.z = Z2;
//
//	float3 Q1;
//	Q1.x = pose.R[0] * Q2.x + pose.R[1] * Q2.y + pose.R[2] * Q2.z + pose.t[0];
//	Q1.y = pose.R[3] * Q2.x + pose.R[4] * Q2.y + pose.R[5] * Q2.z + pose.t[1];
//	Q1.z = pose.R[6] * Q2.x + pose.R[7] * Q2.y + pose.R[8] * Q2.z + pose.t[2];
//
//	// 计算误差 e = n^T * (p1 - q1), E = e^2
//	float ex = P1.x - Q1.x;
//	float ey = P1.y - Q1.y;
//	float ez = P1.z - Q1.z;
//	e_geo = (n.x * ex + n.y * ey + n.z * ez);
//	//if (abs(e) > 3.0f) return;
//
//	// 计算误差对位姿参数的Jacobian（6维：平移vx,vy,vz; 旋转ωx,ωy,ωz）
//	// 平移部分: ∂e/∂v = -n^T
//	Jt_geo[0] = -n.x;  // ∂e/∂vx
//	Jt_geo[1] = -n.y;  // ∂e/∂vy
//	Jt_geo[2] = -n.z;  // ∂e/∂vz
//	// 旋转部分: ∂e/∂ω ≈ -n^T * -[q1]_x (叉积矩阵)
//	// 等价于 Jt[3..5] = (n × q1)
//	Jt_geo[3] = (n.y * Q1.z - n.z * Q1.y); // ∂e/∂ωx
//	Jt_geo[4] = (n.z * Q1.x - n.x * Q1.z); // ∂e/∂ωy
//	Jt_geo[5] = (n.x * Q1.y - n.y * Q1.x); // ∂e/∂ωz
//
//	return true;
//}
//
//
//// 只几何项：计算 g = J^T r （其实你这里是 -J^T r）和 H 的对角
//__global__ void computeGradAndDiagCU(
//    const float* depth1, const float3* norm1,
//    const float* depth2,
//    cuCam cam, const double2* raymap, cuRelPose pose,
//    float* JTr,       // 长度 6
//    float* Hdiag      // 长度 6
//) {
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    if (x >= cam.width || y >= cam.height) return;
//    int idx = y * cam.width + x;
//
//    float e_geo, Jt_geo[6];
//    bool is_geo_ok = computeGeoResidualAndJacobianPixel(
//        x, y, idx, depth1, norm1, depth2,
//        cam, raymap, pose, e_geo, Jt_geo
//    );
//    if (!is_geo_ok) return;
//
//    float w_geo = 1.0f;
//
//    // g = Σ (-w e J)
//    for (int k = 0; k < 6; ++k) {
//        atomicAdd(&JTr[k], -w_geo * e_geo * Jt_geo[k]);
//        // diag(H) ≈ Σ (w J_k^2)
//        atomicAdd(&Hdiag[k], w_geo * Jt_geo[k] * Jt_geo[k]);
//    }
//}
//
//__global__ void zeroVector(float* y, int n) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < n) y[idx] = 0.0f;
//}
//
//__global__ void axpy(float* y, const float* x, float alpha, int n) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < n) y[idx] += alpha * x[idx];
//}
//
//__global__ void applyPreconditionerSingle(
//    const float* r, const float* Minv_diag, float* z, int n)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < n) {
//        z[idx] = Minv_diag[idx] * r[idx];
//    }
//}
//
//// p = beta * p + z
//__global__ void scaleAndAdd(float* __restrict__ p,
//    const float* __restrict__ z,
//    float beta,
//    int n)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx >= n) return;
//
//    p[idx] = beta * p[idx] + z[idx];
//}
//
//__global__ void applyH_singlePair(
//    const float* __restrict__ x,   // 长度 6
//    float* __restrict__ y,         // 长度 6，调用前需置零
//    SinglePairData data)
//{
//    int u = blockIdx.x * blockDim.x + threadIdx.x;
//    int v = blockIdx.y * blockDim.y + threadIdx.y;
//    if (u >= data.cam.width || v >= data.cam.height) return;
//    int idx = v * data.cam.width + u;
//
//    float e_geo;
//    float Jt_geo[6];
//    bool ok = computeGeoResidualAndJacobianPixel(
//        u, v, idx,
//        data.depth1, data.norm1, data.depth2,
//        data.cam, data.raymap, data.pose,
//        e_geo, Jt_geo);
//    if (!ok) return;
//
//    float w_geo = 1.0f;
//
//    // s = w * J * x （标量）
//    float s = 0.0f;
//#pragma unroll
//    for (int k = 0; k < 6; ++k) {
//        s += Jt_geo[k] * x[k];
//    }
//    s *= w_geo;
//
//    // y += J^T * s
//#pragma unroll
//    for (int k = 0; k < 6; ++k) {
//        atomicAdd(&y[k], Jt_geo[k] * s);
//    }
//}
//
//
//void pcgSolveSinglePair(
//    const SinglePairData& data,
//    float* d_delta,   // 长度 6，输出
//    float* d_b,       // 长度 6，输入：b = JTr
//    int maxIters, float tol)
//{
//    const int n = 6;
//    dim3 block1d(64);
//    dim3 grid1d((n + block1d.x - 1) / block1d.x);
//
//    float* d_r, * d_z, * d_p, * d_Ap;
//    cudaMalloc(&d_r, n * sizeof(float));
//    cudaMalloc(&d_z, n * sizeof(float));
//    cudaMalloc(&d_p, n * sizeof(float));
//    cudaMalloc(&d_Ap, n * sizeof(float));
//
//    // x = 0
//    zeroVector << <grid1d, block1d >> > (d_delta, n);
//    // r = b - A x = b
//    cudaMemcpy(d_r, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice);
//
//    // z = M^{-1} r
//    applyPreconditionerSingle << <grid1d, block1d >> > (
//        d_r, data.Minv_diag, d_z, n);
//
//    thrust::device_ptr<float> r_ptr(d_r);
//    thrust::device_ptr<float> z_ptr(d_z);
//    thrust::device_ptr<float> p_ptr(d_p);
//    thrust::device_ptr<float> Ap_ptr(d_Ap);
//
//    // p = z
//    cudaMemcpy(d_p, d_z, n * sizeof(float), cudaMemcpyDeviceToDevice);
//
//    float rz = thrust::inner_product(r_ptr, r_ptr + n, z_ptr, 0.0f);
//
//    // 图像分辨率，用来配置 applyH 的网格
//    dim3 block2d(16, 16);
//    dim3 grid2d(
//        (int(data.cam.width) + block2d.x - 1) / block2d.x,
//        (int(data.cam.height) + block2d.y - 1) / block2d.y);
//
//    for (int iter = 0; iter < maxIters; ++iter) {
//        // Ap = A * p
//        zeroVector << <grid1d, block1d >> > (d_Ap, n);
//        applyH_singlePair << <grid2d, block2d >> > (d_p, d_Ap, data);
//
//        float pAp = thrust::inner_product(p_ptr, p_ptr + n, Ap_ptr, 0.0f);
//        if (fabsf(pAp) < 1e-20f) break;
//        float alpha = rz / pAp;
//
//        // x = x + alpha * p
//        axpy << <grid1d, block1d >> > (d_delta, d_p, alpha, n);
//        // r = r - alpha * Ap
//        axpy << <grid1d, block1d >> > (d_r, d_Ap, -alpha, n);
//
//        // 收敛判断
//        float r2 = thrust::inner_product(r_ptr, r_ptr + n, r_ptr, 0.0f);
//        float rnorm = sqrtf(r2);
//        //printf("[PCG] iter %d, rnorm = %e\n", iter, rnorm);
//        if (rnorm < tol) break;
//
//        // z_new = M^{-1} r
//        applyPreconditionerSingle << <grid1d, block1d >> > (
//            d_r, data.Minv_diag, d_z, n);
//
//        float rz_new = thrust::inner_product(r_ptr, r_ptr + n, z_ptr, 0.0f);
//        float beta = rz_new / rz;
//        rz = rz_new;
//
//        scaleAndAdd << <grid1d, block1d >> > (d_p, d_z, beta, n);
//    }
//
//    cudaFree(d_r);
//    cudaFree(d_z);
//    cudaFree(d_p);
//    cudaFree(d_Ap);
//}
//
//extern "C" void denseOptPoseSE3(
//	cv::Mat mK, cv::Mat mDistCoeffs, cv::Mat Raymap,
//	/*cv::Mat grad1, cv::Mat gradxGrad1, cv::Mat gradyGrad1, cv::Mat depth1, cv::Mat norm1,*/
//	/*cv::Mat grad2,*/
//	vector<shared_ptr<cv::Mat>> mvdepth_KF,
//	vector<shared_ptr<cv::Mat>> mvcolor_KF,
//	vector<shared_ptr<cv::Mat>> mvnormal_KF,
//	vector<Eigen::Matrix4f> mvTwc_KF, vector<Eigen::Matrix4f>&mvTwc_KF_opt, int maxIter = 20, float tol = 1e-6)
//{
//	mvTwc_KF_opt.push_back(mvTwc_KF[0]); // ref frame.
//
//	// Intrinsic stuff
//	int width = Raymap.cols, height = Raymap.rows;
//	size_t imgSize = width * height;
//	cuCam h_cam;
//	h_cam.width = width; h_cam.height = height;
//	h_cam.fx = mK.at<double>(0, 0); h_cam.fy = mK.at<double>(1, 1);
//	h_cam.cx = mK.at<double>(0, 2); h_cam.cy = mK.at<double>(1, 2);
//	h_cam.k1 = static_cast<double>(mDistCoeffs.at<double>(0)); h_cam.k2 = static_cast<double>(mDistCoeffs.at<double>(1)); h_cam.k3 = static_cast<double>(mDistCoeffs.at<double>(4));
//	h_cam.p1 = static_cast<double>(mDistCoeffs.at<double>(2)); h_cam.p2 = static_cast<double>(mDistCoeffs.at<double>(3));
//	double2* d_raymap;
//	cudaMalloc(&d_raymap, imgSize * sizeof(double2)); cudaMemcpy(d_raymap, Raymap.ptr<double>(), imgSize * sizeof(double2), cudaMemcpyHostToDevice);
//
//	Eigen::Matrix4f initPose, optPose;
//	Eigen::Matrix3f R, R_inv;
//	Eigen::Vector3f t, t_inv;
//	cv::Mat depth1, depth2, norm1;
//
//	for (int k = 0; k < 1; ++k)
//	{
//		initPose = mvTwc_KF[k].inverse() * mvTwc_KF[k + 1];
//		initPose.block<3, 1>(0, 3) *= 1.1; // unit: m, test convergence
//		optPose = Eigen::Matrix4f::Identity();
//
//		depth1 = *mvdepth_KF[k];
//		depth2 = *mvdepth_KF[k + 1];
//		//// im1: ring model inner 255, outer 0.
//		//cv::Mat imIi = *mvcolor_KF[j - 1];
//		//cv::Mat imIj = *mvcolor_KF[j];
//		norm1 = *mvnormal_KF[k];
//		//// grad_x_scharr: Scharr(undistorted color)_x.
//		//cv::Mat imGi_x, imGi_y;
//
//		// Map{Intensity, Depth, Normal} stuff
//		/*float* d_grad1; float* d_gradxGrad1; float* d_gradyGrad1;*/ float* d_depth1; float3* d_norm1;
//		/*float* d_grad2;*/ float* d_depth2;
//		/*cudaMalloc(&d_grad1, imgSize * sizeof(float)); cudaMemcpy(d_grad1, grad1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
//		cudaMalloc(&d_gradxGrad1, imgSize * sizeof(float)); cudaMemcpy(d_gradxGrad1, gradxGrad1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
//		cudaMalloc(&d_gradyGrad1, imgSize * sizeof(float)); cudaMemcpy(d_gradyGrad1, gradyGrad1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);*/
//		cudaMalloc(&d_depth1, imgSize * sizeof(float)); cudaMemcpy(d_depth1, depth1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
//		cudaMalloc(&d_norm1, imgSize * sizeof(float3)); cudaMemcpy(d_norm1, norm1.ptr<float>(), imgSize * sizeof(float3), cudaMemcpyHostToDevice);
//		/*cudaMalloc(&d_grad2, imgSize * sizeof(float)); cudaMemcpy(d_grad2, grad2.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);*/
//		cudaMalloc(&d_depth2, imgSize * sizeof(float)); cudaMemcpy(d_depth2, depth2.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
//
//		float* d_JTr_single;   // 长度 6
//		float* d_Hdiag_single; // 长度 6
//		float* d_Minv_single;  // 长度 6
//		float* d_delta_single; // 长度 6
//
//		cudaMalloc(&d_JTr_single, 6 * sizeof(float));
//		cudaMalloc(&d_Hdiag_single, 6 * sizeof(float));
//		cudaMalloc(&d_Minv_single, 6 * sizeof(float));
//		cudaMalloc(&d_delta_single, 6 * sizeof(float));
//
//		// CUDA线程块与网格配置
//		dim3 threads(16, 16);
//		dim3 blocks((width + 15) / 16, (height + 15) / 16);
//
//		// 将初始位姿拆分为旋转和平移
//		R = initPose.block<3, 3>(0, 0); R_inv = R.transpose();
//		t = initPose.block<3, 1>(0, 3); t_inv = t_inv = -R_inv * t;
//
//		cout << "INIT: " << initPose << endl;
//
//        for (int iter = 0; iter < maxIter; ++iter)
//        {
//            if (t.norm() < 1e-4) continue;
//
//            // === 1) 计算 g 和 diag(H) ===
//            cudaMemset(d_JTr_single, 0, 6 * sizeof(float));
//            cudaMemset(d_Hdiag_single, 0, 6 * sizeof(float));
//
//            cuRelPose h_pose;
//            for (int i = 0; i < 3; ++i) {
//                for (int j = 0; j < 3; ++j) {
//                    h_pose.R[i * 3 + j] = R(i, j);
//                    h_pose.R_inv[i * 3 + j] = R_inv(i, j);
//                }
//                h_pose.t[i] = t[i];
//                h_pose.t_inv[i] = t_inv[i];
//            }
//
//            computeGradAndDiagCU << <blocks, threads >> > (
//                d_depth1, d_norm1, d_depth2,
//                h_cam, d_raymap, h_pose,
//                d_JTr_single, d_Hdiag_single);
//
//            cudaDeviceSynchronize();
//
//            // debug for global PCG
//            float h_JTr[6];
//            cudaMemcpy(h_JTr, d_JTr_single, 6 * sizeof(float), cudaMemcpyDeviceToHost);
//            cout << "Jtr: ";
//            for (int i = 0; i < 6; ++i) { cout << h_JTr[i] << "\t"; }
//            cout << endl;
//
//            // === 2) 构造 Jacobi 预条件器 M^{-1} ===
//            float h_Hdiag[6];
//            cudaMemcpy(h_Hdiag, d_Hdiag_single, 6 * sizeof(float), cudaMemcpyDeviceToHost);
//
//            float h_Minv[6];
//            for (int i = 0; i < 6; ++i) {
//                if (h_Hdiag[i] > 1e-8f)
//                    h_Minv[i] = 1.0f / h_Hdiag[i];
//                else
//                    h_Minv[i] = 1.0f; // 或者给个大的值，看情况调
//            }
//            cudaMemcpy(d_Minv_single, h_Minv, 6 * sizeof(float), cudaMemcpyHostToDevice);
//
//            // === 3) 调 PCG 解 H δ = b（b = JTr_single） ===
//            SinglePairData spData;
//            spData.cam = h_cam;
//            spData.depth1 = d_depth1;
//            spData.norm1 = d_norm1;
//            spData.depth2 = d_depth2;
//            spData.raymap = d_raymap;
//            spData.pose = h_pose;
//            spData.Minv_diag = d_Minv_single;
//
//            pcgSolveSinglePair(
//                spData,
//                d_delta_single,
//                d_JTr_single,
//                /*maxItersCG*/ 20,
//                /*tolCG*/ 1e-8f);
//
//            // 拷回 δξ
//            float h_delta[6];
//            cudaMemcpy(h_delta, d_delta_single, 6 * sizeof(float), cudaMemcpyDeviceToHost);
//
//            Eigen::Matrix<float, 6, 1> delta;
//            cout << "delta: ";
//            for (int i = 0; i < 6; ++i) { delta(i) = h_delta[i]; cout << h_delta[i] << "\t"; }
//            cout << endl;
//
//            if (delta.norm() < tol) break;
//
//            // === 4) 和原来一样，用 δ 更新 R, t ===
//            Eigen::Vector3f omega = delta.tail<3>();
//            Eigen::Vector3f upsilon = delta.head<3>();
//
//            float theta = omega.norm();
//            Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
//            if (theta > 1e-12) {
//                Eigen::Vector3f k = omega / theta;
//                Eigen::Matrix3f K;
//                K << 0, -k(2), k(1),
//                    k(2), 0, -k(0),
//                    -k(1), k(0), 0;
//                dR = Eigen::Matrix3f::Identity()
//                    + sin(theta) * K
//                    + (1.0f - cos(theta)) * K * K;
//            }
//            Eigen::Vector3f dt = upsilon;
//
//            R = dR * R;
//            t = dR * t + dt;
//
//            R_inv = R.transpose();
//            t_inv = -R_inv * t;
//
//            optPose.block<3, 3>(0, 0) = R;
//            optPose.block<3, 1>(0, 3) = t; // uint: m
//            cout << "OPTI: " << optPose << endl;
//        }
//
//		// 释放GPU内存
//		/*cudaFree(d_grad1); cudaFree(d_gradxGrad1); cudaFree(d_gradyGrad1);*/ cudaFree(d_depth1); cudaFree(d_norm1);
//		/*cudaFree(d_grad2);*/ cudaFree(d_depth2);
//        cudaFree(d_JTr_single);
//        cudaFree(d_Hdiag_single);
//        cudaFree(d_Minv_single);
//        cudaFree(d_delta_single);
//
//		// 返回优化后的位姿矩阵
//		Eigen::Matrix4f Twc_KF_opt = mvTwc_KF_opt[k] * optPose;
//		mvTwc_KF_opt.push_back(Twc_KF_opt);
//	}
//	cudaFree(d_raymap);
//}

////__global__ void computeErrorAndJacobian_dense(
////	/*const float* grad1, const float* gradxGrad1, const float* gradyGrad1,*/
////	const float* depth1, const float3* norm1, /*const float* grad2,*/ const float* depth2,
////	int width, int height, double fx, double fy, double cx, double cy,
////	const float* R, const float* t, const float* R_inv, const float* t_inv,
////	float* JTJ, float* JTr, float* d_eMap)
////{
////	int x = blockIdx.x * blockDim.x + threadIdx.x;
////	int y = blockIdx.y * blockDim.y + threadIdx.y;
////	if (x >= width || y >= height) return;
////	int idx = y * width + x;
////
////	// ---- ---- ---- ---- ---- geometric term ---- ---- ---- ---- ---- //
////	bool is_geo_ok = true;
////	float w_geo = 1;
////
////	float Z1 = depth1[idx];
////	if (Z1 <= 0) is_geo_ok = false;
////	// 重建源帧中像素点的3D坐标 p1
////	float X1 = (x - cx) * Z1 / fx;
////	float Y1 = (y - cy) * Z1 / fy;
////	float3 p1 = make_float3(X1, Y1, Z1);
////
////	// 将点变换到目标帧坐标系 p2 = T_inv * P1 = R_inv * p1 + t_inv
////	float3 p2;
////	p2.x = R_inv[0] * p1.x + R_inv[1] * p1.y + R_inv[2] * p1.z + t_inv[0];
////	p2.y = R_inv[3] * p1.x + R_inv[4] * p1.y + R_inv[5] * p1.z + t_inv[1];
////	p2.z = R_inv[6] * p1.x + R_inv[7] * p1.y + R_inv[8] * p1.z + t_inv[2];
////
////	float u2 = (p2.x * fx) / p2.z + cx;
////	float v2 = (p2.y * fy) / p2.z + cy;
////	int u2_d = __float2int_rd(u2); int v2_d = __float2int_rd(v2); // => floor()
////	int u2_u = __float2int_ru(u2); int v2_u = __float2int_ru(v2); // => ceil()
////	if (u2_d < 0 || u2_u >= width || v2_d < 0 || v2_u >= height) is_geo_ok = false;
////	int idx2_dd = v2_d * width + u2_d; float Z2_dd = depth2[idx2_dd];
////	int idx2_du = v2_u * width + u2_d; float Z2_du = depth2[idx2_du];
////	int idx2_ud = v2_d * width + u2_u; float Z2_ud = depth2[idx2_ud];
////	if (Z2_dd == 0 || Z2_du == 0 || Z2_ud == 0) is_geo_ok = false;
////	float Z2 = Z2_dd + (u2 - u2_d) * (Z2_ud - Z2_dd) + (v2 - v2_d) * (Z2_du - Z2_dd);
////	if (abs(Z2 - p2.z) > 3.0f) is_geo_ok = false; // occlusion between objects
////
////	// 重建目标帧对应点 q 和法线 n
////	float3 q2;
////	q2.x = (u2 - cx) * Z2 / fx;
////	q2.y = (v2 - cy) * Z2 / fy;
////	q2.z = Z2;
////	float3 q1;
////	q1.x = R[0] * q2.x + R[1] * q2.y + R[2] * q2.z + t[0];
////	q1.y = R[3] * q2.x + R[4] * q2.y + R[5] * q2.z + t[1];
////	q1.z = R[6] * q2.x + R[7] * q2.y + R[8] * q2.z + t[2];
////
////	float3 n = norm1[idx];
////
////	// 计算误差 e = n^T * (p1 - q1), E = e^2
////	float ex = p1.x - q1.x;
////	float ey = p1.y - q1.y;
////	float ez = p1.z - q1.z;
////	float e_geo = (n.x * ex + n.y * ey + n.z * ez);
////	//if (abs(e) > 3.0f) return;
////
////	// 计算误差对位姿参数的Jacobian（6维：平移vx,vy,vz; 旋转ωx,ωy,ωz）
////	// 平移部分: ∂e/∂v = -n^T
////	float Jt_geo[6];
////	Jt_geo[0] = -n.x;  // ∂e/∂vx
////	Jt_geo[1] = -n.y;  // ∂e/∂vy
////	Jt_geo[2] = -n.z;  // ∂e/∂vz
////	// 旋转部分: ∂e/∂ω ≈ -n^T * -[q1]_x (叉积矩阵)
////	// 等价于 Jt[3..5] = (n × q1)
////	float px = q1.x, py = q1.y, pz = q1.z;
////	Jt_geo[3] = (n.y * pz - n.z * py); // ∂e/∂ωx
////	Jt_geo[4] = (n.z * px - n.x * pz); // ∂e/∂ωy
////	Jt_geo[5] = (n.x * py - n.y * px); // ∂e/∂ωz
////
////	/*
////	// ---- ---- ---- ---- ---- photometric term ---- ---- ---- ---- ---- //
////	bool is_pho_ok = true;
////	float alpha_pho = 25.0f;
////	float w_pho = 2e-2 / 3e2;
////
////	if (x < 20 || x >= width - 20 || y < 20 || y >= height - 20) is_pho_ok = false;
////	float I2 = grad2[idx];
////	if (I2 >= 2e3) is_pho_ok = false;
////	float Z02 = depth2[idx];
////	if (Z02 <= 0) is_pho_ok = false;
////
////	float X2 = (x - cx) * Z02 / fx;
////	float Y2 = (y - cy) * Z02 / fy;
////	float3 p02 = make_float3(X2, Y2, Z02);
////
////	float3 p01;
////	p01.x = R[0] * p02.x + R[1] * p02.y + R[2] * p02.z + t[0];
////	p01.y = R[3] * p02.x + R[4] * p02.y + R[5] * p02.z + t[1];
////	p01.z = R[6] * p02.x + R[7] * p02.y + R[8] * p02.z + t[2];
////
////	float u1 = (p01.x * fx) / p01.z + cx;
////	float v1 = (p01.y * fy) / p01.z + cy;
////	int u1_d = __float2int_rd(u1); int v1_d = __float2int_rd(v1); // => floor()
////	int u1_u = __float2int_ru(u1); int v1_u = __float2int_ru(v1); // => ceil()
////	int u1_n = __float2int_rn(u1); int v1_n = __float2int_rn(v1);
////	if (u1_d < 20 || u1_u >= width - 20 || v1_d < 20 || v1_u >= height - 20) is_pho_ok = false;
////	int idx1_dd = v1_d * width + u1_d; float I1_dd = grad1[idx1_dd];
////	int idx1_du = v1_u * width + u1_d; float I1_du = grad1[idx1_du];
////	int idx1_ud = v1_d * width + u1_u; float I1_ud = grad1[idx1_ud];
////	int idx1_nn = v1_n * width + u1_n;
////	float I1 = I1_dd + (u1 - u1_d) * (I1_ud - I1_dd) + (v1 - v1_d) * (I1_du - I1_dd);
////
////	float e_pho = (I2 - I1) * alpha_pho;
////	//if (abs(e) > 2e2f) return;
////
////	// 计算误差对位姿参数的Jacobian（6维：平移vx,vy,vz; 旋转ωx,ωy,ωz）
////	float ggx = gradxGrad1[idx1_nn];
////	float ggy = gradyGrad1[idx1_nn];
////	float Jt_pho[6];
////	// 平移部分: ∂e/∂v = -∇I1(u1) * ∂π/∂p1 * E
////	Jt_pho[0] = -ggx * (fx / p01.z);  // ∂e/∂vx
////	Jt_pho[1] = -ggy * (fy / p01.z);  // ∂e/∂vy
////	Jt_pho[2] = -(Jt_pho[0] * p01.x + Jt_pho[1] * p01.y) / p01.z;  // ∂e/∂vz
////	// 旋转部分: ∂e/∂ω ≈ -∇I1(u1) * ∂π/∂p1 * -[p1]_x (叉积矩阵)
////	Jt_pho[3] = p01.y * Jt_pho[2] + fy * ggy; // ∂e/∂ωx
////	Jt_pho[4] = -p01.x * Jt_pho[2] - fx * ggx; // ∂e/∂ωy
////	Jt_pho[5] = -Jt_pho[0] * p01.y + Jt_pho[1] * p01.x; // ∂e/∂ωz
////	*/
////
////	// ---- ---- ---- ---- ---- add up ---- ---- ---- ---- ---- //
////	if (is_geo_ok) {
////		d_eMap[idx] = d_eMap[idx] + e_geo;
////		// 原子累加到全局JTJ和JTr
////		for (int i = 0; i < 6; ++i) {
////			atomicAdd(&JTr[i], -w_geo * e_geo * Jt_geo[i]);  // J^T * r (这里累加 -e*Jt)
////			for (int j = 0; j <= i; ++j) {
////				atomicAdd(&JTJ[i * 6 + j], w_geo * Jt_geo[i] * Jt_geo[j]);
////				if (i != j) {
////					atomicAdd(&JTJ[j * 6 + i], w_geo * Jt_geo[i] * Jt_geo[j]);
////				}
////			}
////		}
////	}
////	/*if (is_pho_ok) {
////		d_eMap[idx] = d_eMap[idx] + e_pho / 2e3;
////		// 原子累加到全局JTJ和JTr
////		for (int i = 0; i < 6; ++i) {
////			atomicAdd(&JTr[i], -w_pho * e_pho * Jt_pho[i]);  // J^T * r (这里累加 -e*Jt)
////			for (int j = 0; j <= i; ++j) {
////				atomicAdd(&JTJ[i * 6 + j], w_pho * Jt_pho[i] * Jt_pho[j]);
////				if (i != j) {
////					atomicAdd(&JTJ[j * 6 + i], w_pho * Jt_pho[i] * Jt_pho[j]);
////				}
////			}
////		}
////	}*/
////}
////inline int denseOptImpl()
////{
////	const string pathToPoses = R"(D:\ins_att\ORB_SLAM3-master\bin\KeyFrameTrajectory.txt)";
////	//const string pathToPoses = R"(C:\Users\ST23re\Desktop\sparse_refined.txt)";
////	const string pathToSrc = R"(D:\ins_att\MatteSpot\data\dataset)";
////
////	const unsigned int w = 1120, h = 800;
////	double fx = 1154.93, fy = 1154.95, cx = 560.74, cy = 400.553;
////	double k1 = -0.0867707, k2 = 0.147489, k3 = -0.0370996;
////	double p1 = 0.000417617, p2 = 0.000554355;
////	const cv::Mat mK = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
////	const cv::Mat mDistCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
////
////	cv::Mat map1, map2;
////	cv::initUndistortRectifyMap(
////		mK, mDistCoeffs,			// 相机内参与畸变系数
////		cv::Mat::eye(3, 3, CV_64F), // 复位变换矩阵 (identity 或 R)
////		mK,							// 校正后的投影矩阵 (可与 K 相同)
////		cv::Size(w, h),             // 图像尺寸
////		CV_32FC1,					// 输出 map 类型
////		map1, map2);
////
////	// -------- prefix end --------
////	std::vector<Eigen::Matrix4f> poses, poses_opt;
////	//if (!loadCameraPoses(pathToPoses, poses)) return -1;
////	poses_opt = poses;
////	vector<cv::Mat> imGs, imDs, imNs;
////	//if (!loadRGBDNormal(pathToSrc, map1, map2, imGs, imDs, imNs)) return -1;
////
////	size_t nframes = poses.size();
////	if (nframes != imGs.size()) { cout << "Inconsistent Input!" << endl; return -1; }
////	else cout << nframes << "frames Loaded!" << endl;
////	vector<cv::Point3f> points;
////	vector<cv::Vec3b> colors;
////	for (int i = 0; i < nframes - 1; i++)
////	{
////		// initial pose => T_ij = inv(T_i) * T_j
////		Eigen::Matrix4f T_i = poses[i], T_j = poses[i + 1];
////		Eigen::Matrix4f T_ij = T_i.inverse() * T_j;
////		//T_ij.block<3, 1>(0, 3) *= 1.01; // test pho converge
////
////		// optimize
////		cv::Mat im1, im2;
////		//im1 = ExtractMatteSpots(imGs[i]);
////		cv::Mat grad_x_scharr, grad_y_scharr;
////		cv::Scharr(im1, grad_x_scharr, CV_32F, 1, 0); cv::blur(grad_x_scharr, grad_x_scharr, cv::Size(3, 3));
////		cv::Scharr(im1, grad_y_scharr, CV_32F, 0, 1); cv::blur(grad_y_scharr, grad_y_scharr, cv::Size(3, 3));
////		//im2 = ExtractMatteSpots(imGs[i + 1]);
////
////		Eigen::Matrix4f T_opt;
////		denseOptPoseSE3(
////			im1, grad_x_scharr, grad_y_scharr, imDs[i], imNs[i],
////			im2, imDs[i + 1],
////			T_ij, T_opt
////		);
////
////		// update
////		poses_opt[i + 1] = poses_opt[i] * T_opt;
////
////		//// render
////		//if (i == 0) textureDepthTo3D(imGs[i], imDs[i], poses_opt[0], points, colors);
////		//textureDepthTo3D(imGs[i + 1], imDs[i + 1], poses_opt[i + 1], points, colors);
////	}
////
////	// 输出优化后的位姿序列
////	std::ofstream fout("optimized_poses.txt");
////	for (const auto& T : poses_opt) {
////		for (int r = 0; r < 4; ++r) {
////			for (int c = 0; c < 4; ++c) {
////				fout << T(r, c) << " ";
////			}
////			fout << "\n";
////		}
////	}
////	fout.close();
////
////	//saveModeltoPly(points, colors);
////
////	return 0;
////}
//
//__global__ void computeErrorAndJacobianCU(
//	/*const float* grad1, const float* gradxGrad1, const float* gradyGrad1,*/
//	const float* depth1, const float3* norm1, /*const float* grad2,*/ const float* depth2,
//	cuCam cam, const double2* raymap, cuPose pose,
//	float* JTJ, float* JTr, float* d_eMap)
//{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//	if (x >= cam.width || y >= cam.height) return;
//	int idx = y * cam.width + x;
//
//	// ---- ---- ---- ---- ---- geometric term ---- ---- ---- ---- ---- //
//	float w_geo = 1;
//	float e_geo, Jt_geo[6];
//	bool is_geo_ok = computeGeoResidualAndJacobianPixel(
//		x, y, idx, depth1, norm1, depth2,
//		cam, raymap, pose, e_geo, Jt_geo
//	);
//
//	/*
//	// ---- ---- ---- ---- ---- photometric term ---- ---- ---- ---- ---- //
//	bool is_pho_ok = true;
//	float alpha_pho = 25.0f;
//	float w_pho = 2e-2 / 3e2;
//
//	if (x < 20 || x >= width - 20 || y < 20 || y >= height - 20) is_pho_ok = false;
//	float I2 = grad2[idx];
//	if (I2 >= 2e3) is_pho_ok = false;
//	float Z02 = depth2[idx];
//	if (Z02 <= 0) is_pho_ok = false;
//
//	float X2 = (x - cx) * Z02 / fx;
//	float Y2 = (y - cy) * Z02 / fy;
//	float3 p02 = make_float3(X2, Y2, Z02);
//
//	float3 p01;
//	p01.x = R[0] * p02.x + R[1] * p02.y + R[2] * p02.z + t[0];
//	p01.y = R[3] * p02.x + R[4] * p02.y + R[5] * p02.z + t[1];
//	p01.z = R[6] * p02.x + R[7] * p02.y + R[8] * p02.z + t[2];
//
//	float u1 = (p01.x * fx) / p01.z + cx;
//	float v1 = (p01.y * fy) / p01.z + cy;
//	int u1_d = __float2int_rd(u1); int v1_d = __float2int_rd(v1); // => floor()
//	int u1_u = __float2int_ru(u1); int v1_u = __float2int_ru(v1); // => ceil()
//	int u1_n = __float2int_rn(u1); int v1_n = __float2int_rn(v1);
//	if (u1_d < 20 || u1_u >= width - 20 || v1_d < 20 || v1_u >= height - 20) is_pho_ok = false;
//	int idx1_dd = v1_d * width + u1_d; float I1_dd = grad1[idx1_dd];
//	int idx1_du = v1_u * width + u1_d; float I1_du = grad1[idx1_du];
//	int idx1_ud = v1_d * width + u1_u; float I1_ud = grad1[idx1_ud];
//	int idx1_nn = v1_n * width + u1_n;
//	float I1 = I1_dd + (u1 - u1_d) * (I1_ud - I1_dd) + (v1 - v1_d) * (I1_du - I1_dd);
//
//	float e_pho = (I2 - I1) * alpha_pho;
//	//if (abs(e) > 2e2f) return;
//
//	// 计算误差对位姿参数的Jacobian（6维：平移vx,vy,vz; 旋转ωx,ωy,ωz）
//	float ggx = gradxGrad1[idx1_nn];
//	float ggy = gradyGrad1[idx1_nn];
//	float Jt_pho[6];
//	// 平移部分: ∂e/∂v = -∇I1(u1) * ∂π/∂p1 * E
//	Jt_pho[0] = -ggx * (fx / p01.z);  // ∂e/∂vx
//	Jt_pho[1] = -ggy * (fy / p01.z);  // ∂e/∂vy
//	Jt_pho[2] = -(Jt_pho[0] * p01.x + Jt_pho[1] * p01.y) / p01.z;  // ∂e/∂vz
//	// 旋转部分: ∂e/∂ω ≈ -∇I1(u1) * ∂π/∂p1 * -[p1]_x (叉积矩阵)
//	Jt_pho[3] = p01.y * Jt_pho[2] + fy * ggy; // ∂e/∂ωx
//	Jt_pho[4] = -p01.x * Jt_pho[2] - fx * ggx; // ∂e/∂ωy
//	Jt_pho[5] = -Jt_pho[0] * p01.y + Jt_pho[1] * p01.x; // ∂e/∂ωz
//	*/
//
//	// ---- ---- ---- ---- ---- add up ---- ---- ---- ---- ---- //
//	if (is_geo_ok)
//	{
//		d_eMap[idx] = d_eMap[idx] + e_geo;
//		// 原子累加到全局JTJ和JTr
//		for (int i = 0; i < 6; ++i) {
//			atomicAdd(&JTr[i], -w_geo * e_geo * Jt_geo[i]);  // J^T * r (这里累加 -e*Jt)
//			for (int j = 0; j <= i; ++j) {
//				atomicAdd(&JTJ[i * 6 + j], w_geo * Jt_geo[i] * Jt_geo[j]);
//				if (i != j) {
//					atomicAdd(&JTJ[j * 6 + i], w_geo * Jt_geo[i] * Jt_geo[j]);
//				}
//			}
//		}
//	}
//	/*
//	if (is_pho_ok) {
//		d_eMap[idx] = d_eMap[idx] + e_pho / 2e3;
//		// 原子累加到全局JTJ和JTr
//		for (int i = 0; i < 6; ++i) {
//			atomicAdd(&JTr[i], -w_pho * e_pho * Jt_pho[i]);  // J^T * r (这里累加 -e*Jt)
//			for (int j = 0; j <= i; ++j) {
//				atomicAdd(&JTJ[i * 6 + j], w_pho * Jt_pho[i] * Jt_pho[j]);
//				if (i != j) {
//					atomicAdd(&JTJ[j * 6 + i], w_pho * Jt_pho[i] * Jt_pho[j]);
//				}
//			}
//		}
//	}*/
//}
//
//extern "C" void denseOptPoseSE3(
//	cv::Mat mK, cv::Mat mDistCoeffs, cv::Mat Raymap,
//	/*cv::Mat grad1, cv::Mat gradxGrad1, cv::Mat gradyGrad1, cv::Mat depth1, cv::Mat norm1,*/
//	/*cv::Mat grad2,*/
//	vector<shared_ptr<cv::Mat>> mvdepth_KF,
//	vector<shared_ptr<cv::Mat>> mvcolor_KF,
//	vector<shared_ptr<cv::Mat>> mvnormal_KF,
//	vector<Eigen::Matrix4f> mvTwc_KF, vector<Eigen::Matrix4f>&mvTwc_KF_opt, int maxIter = 20, float tol = 1e-6)
//{
//	mvTwc_KF_opt.push_back(mvTwc_KF[0]); // ref frame.
//
//	// Intrinsic stuff
//	int width = Raymap.cols, height = Raymap.rows;
//	size_t imgSize = width * height;
//	cuCam h_cam;
//	h_cam.width = width; h_cam.height = height;
//	h_cam.fx = mK.at<double>(0, 0); h_cam.fy = mK.at<double>(1, 1);
//	h_cam.cx = mK.at<double>(0, 2); h_cam.cy = mK.at<double>(1, 2);
//	h_cam.k1 = static_cast<double>(mDistCoeffs.at<double>(0)); h_cam.k2 = static_cast<double>(mDistCoeffs.at<double>(1)); h_cam.k3 = static_cast<double>(mDistCoeffs.at<double>(4));
//	h_cam.p1 = static_cast<double>(mDistCoeffs.at<double>(2)); h_cam.p2 = static_cast<double>(mDistCoeffs.at<double>(3));
//	double2* d_raymap;
//	cudaMalloc(&d_raymap, imgSize * sizeof(double2)); cudaMemcpy(d_raymap, Raymap.ptr<double>(), imgSize * sizeof(double2), cudaMemcpyHostToDevice);
//
//	Eigen::Matrix4f initPose, optPose;
//	Eigen::Matrix3f R, R_inv;
//	Eigen::Vector3f t, t_inv;
//	cv::Mat depth1, depth2, norm1;
//
//	for (int k = 0; k < mvdepth_KF.size() - 1; ++k)
//	{
//		initPose = mvTwc_KF[k].inverse() * mvTwc_KF[k + 1];
//		initPose.block<3, 1>(0, 3) *= 1000.0f; // unit: mm
//		optPose = Eigen::Matrix4f::Identity();
//
//		depth1 = *mvdepth_KF[k];
//		depth2 = *mvdepth_KF[k + 1];
//		//// im1: ring model inner 255, outer 0.
//		//cv::Mat imIi = *mvcolor_KF[j - 1];
//		//cv::Mat imIj = *mvcolor_KF[j];
//		norm1 = *mvnormal_KF[k];
//		//// grad_x_scharr: Scharr(undistorted color)_x.
//		//cv::Mat imGi_x, imGi_y;
//
//		// Map{Intensity, Depth, Normal} stuff
//		/*float* d_grad1; float* d_gradxGrad1; float* d_gradyGrad1;*/ float* d_depth1; float3* d_norm1;
//		/*float* d_grad2;*/ float* d_depth2;
//		/*cudaMalloc(&d_grad1, imgSize * sizeof(float)); cudaMemcpy(d_grad1, grad1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
//		cudaMalloc(&d_gradxGrad1, imgSize * sizeof(float)); cudaMemcpy(d_gradxGrad1, gradxGrad1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
//		cudaMalloc(&d_gradyGrad1, imgSize * sizeof(float)); cudaMemcpy(d_gradyGrad1, gradyGrad1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);*/
//		cudaMalloc(&d_depth1, imgSize * sizeof(float)); cudaMemcpy(d_depth1, depth1.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
//		cudaMalloc(&d_norm1, imgSize * sizeof(float3)); cudaMemcpy(d_norm1, norm1.ptr<float>(), imgSize * sizeof(float3), cudaMemcpyHostToDevice);
//		/*cudaMalloc(&d_grad2, imgSize * sizeof(float)); cudaMemcpy(d_grad2, grad2.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);*/
//		cudaMalloc(&d_depth2, imgSize * sizeof(float)); cudaMemcpy(d_depth2, depth2.ptr<float>(), imgSize * sizeof(float), cudaMemcpyHostToDevice);
//
//		// debug: check error map
//		float* d_eMap;
//		cudaMalloc(&d_eMap, imgSize * sizeof(float));
//
//		// 为JTJ和JTr分配GPU内存
//		float* d_JTr;  /*梯度向量6dof*/ float* d_JTJ;  /* Hessian 矩阵 (6x6 = 36)*/
//		cudaMalloc(&d_JTr, 6 * sizeof(float));
//		cudaMalloc(&d_JTJ, 36 * sizeof(float));
//
//		// CUDA线程块与网格配置
//		dim3 threads(16, 16);
//		dim3 blocks((width + 15) / 16, (height + 15) / 16);
//
//		// 将初始位姿拆分为旋转和平移
//		R = initPose.block<3, 3>(0, 0); R_inv = R.transpose();
//		t = initPose.block<3, 1>(0, 3); t_inv = t_inv = -R_inv * t;
//
//		cout << "INIT: " << initPose << endl;
//
//		for (int iter = 0; iter < maxIter; ++iter)
//		{
//			if (t.norm() < 1e-4) continue; // WTF
//
//			// 清零JTJ和JTr
//			cudaMemset(d_JTr, 0, 6 * sizeof(float));
//			cudaMemset(d_JTJ, 0, 36 * sizeof(float));
//			cudaMemset(d_eMap, 0, imgSize * sizeof(float)); // debug
//
//			// 准备当前旋转矩阵和平移向量数据
//			cuPose h_pose;
//			for (int i = 0; i < 3; i++) {
//				for (int j = 0; j < 3; j++) {
//					h_pose.R[i * 3 + j] = R(i, j);
//					h_pose.R_inv[i * 3 + j] = R_inv(i, j);
//				}
//				h_pose.t[i] = t[i];
//				h_pose.t_inv[i] = t_inv[i];
//			}
//
//			// 调用CUDA核函数计算误差和Jacobian累积
//			computeErrorAndJacobianCU << <blocks, threads >> > (
//				/*d_grad1, d_gradxGrad1, d_gradyGrad1,*/ d_depth1, d_norm1, /*d_grad2,*/ d_depth2,
//				h_cam, d_raymap, h_pose,
//				d_JTJ, d_JTr,
//				d_eMap
//				);
//
//			cudaError_t cudaStatus;
//			// Check for any errors launching the kernel
//			cudaStatus = cudaGetLastError();
//			if (cudaStatus != cudaSuccess) {
//				fprintf(stderr, "computeErrorAndJacobianCU launch failed: %s\n", cudaGetErrorString(cudaStatus));
//				system("pause");
//			}
//			cudaStatus = cudaDeviceSynchronize();
//			if (cudaStatus != cudaSuccess) {
//				fprintf(stderr, "cudaDeviceSynchronize launch failed: %s\n", cudaGetErrorString(cudaStatus));
//				system("pause");
//			}
//
//			// 将累积结果从GPU拷贝回CPU
//			float h_JTJ[36]; float h_JTr[6]; cv::Mat h_eMap(height, width, CV_32F);
//			cudaMemcpy(h_JTJ, d_JTJ, 36 * sizeof(float), cudaMemcpyDeviceToHost);
//			cudaMemcpy(h_JTr, d_JTr, 6 * sizeof(float), cudaMemcpyDeviceToHost);
//			cudaMemcpy(h_eMap.ptr<float>(), d_eMap, imgSize * sizeof(float), cudaMemcpyDeviceToHost); // debug
//
//			// 构造Eigen矩阵并求解增量 Δξ = (J^T J)^{-1} (-J^T r)
//			Eigen::Matrix<float, 6, 6> H;
//			Eigen::Matrix<float, 6, 1> g;
//			for (int i = 0; i < 6; ++i) {
//				g(i) = h_JTr[i];
//				for (int j = 0; j < 6; ++j) {
//					H(i, j) = h_JTJ[i * 6 + j];
//				}
//			}
//			Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(g);
//			// 检查收敛条件
//			if (delta.norm() < tol) break;
//
//			// 构造SE(3)增量变换：旋转部分使用Rodrigues公式
//			Eigen::Vector3f omega = delta.tail<3>();
//			Eigen::Vector3f upsilon = delta.head<3>();
//			// Rodrigues: 旋转向量 -> 3x3 矩阵
//			cv::Mat rvec = (cv::Mat_<float>(3, 1) << omega(0), omega(1), omega(2));
//			cv::Mat Rinc_cv;
//			cv::Rodrigues(rvec, Rinc_cv);
//			Eigen::Matrix3f Rinc;
//			Rinc << Rinc_cv.at<float>(0, 0), Rinc_cv.at<float>(0, 1), Rinc_cv.at<float>(0, 2),
//				Rinc_cv.at<float>(1, 0), Rinc_cv.at<float>(1, 1), Rinc_cv.at<float>(1, 2),
//				Rinc_cv.at<float>(2, 0), Rinc_cv.at<float>(2, 1), Rinc_cv.at<float>(2, 2);
//			// 更新当前位姿：左乘增量变换
//			R = Rinc * R;
//			t = Rinc * t + upsilon;
//			R_inv = R.transpose();
//			t_inv = -R_inv * t;
//
//			//// debug
//			//cout << "Iter[" << iter << "] delta: " << delta.norm() << endl;
//			////cv::Mat errorMat(height, width, CV_32FC1, h_eMap.data());
//			////cv::imwrite("error_map.tif", errorMat);
//			//cv::Mat absMat(height, width, CV_32F), validMat(height, width, CV_8U), display(height, width, CV_8UC3);
//			//absMat = abs(h_eMap);
//			//validMat = absMat > 0;
//			//validMat /= 255;
//			//cout << "Average pixel error: " << cv::sum(absMat)[0] / cv::sum(validMat)[0] << "mm" << endl;
//			//cout << "Valid pixel count: " << cv::sum(validMat)[0] << " s" << endl;
//			//cv::normalize(absMat, absMat, 0, 255, cv::NORM_MINMAX, CV_8U);
//			//cv::applyColorMap(absMat, display, cv::COLORMAP_JET);
//			//cv::imshow("error map from cuda", display);
//			//cv::waitKey(1);
//
//			//system("pause");
//		}
//
//		// 释放GPU内存
//		/*cudaFree(d_grad1); cudaFree(d_gradxGrad1); cudaFree(d_gradyGrad1);*/ cudaFree(d_depth1); cudaFree(d_norm1);
//		/*cudaFree(d_grad2);*/ cudaFree(d_depth2);
//		cudaFree(d_JTJ);
//		cudaFree(d_JTr);
//		cudaFree(d_eMap);
//
//		// 返回优化后的位姿矩阵
//		optPose.block<3, 3>(0, 0) = R;
//		optPose.block<3, 1>(0, 3) = t * 1e-3; // uint: m
//		Eigen::Matrix4f Twc_KF_opt = mvTwc_KF_opt[k] * optPose;
//		mvTwc_KF_opt.push_back(Twc_KF_opt);
//
//		cout << "OPTI: " << optPose << endl;
//	}
//	cudaFree(d_raymap);
//}