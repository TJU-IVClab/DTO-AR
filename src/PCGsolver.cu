//#include "DenseOptimizer.cuh"
//#include "KeyFrame.h"
//
//#include <thrust/device_ptr.h>
//#include <thrust/inner_product.h>
//
//inline __device__ float2 reproject_ideal_to_distorted(const float2& p_ideal_center, cuCam cam)
//{
//    // ====================================================
//    //  Ideal pixel coordinates -> Normalized coordinates (xn, yn) [Double calculation]
//    // ====================================================
//    double xn = (/*static_cast<double>*/(p_ideal_center.x) - cam.cx) / cam.fx;
//    double yn = (/*static_cast<double>*/(p_ideal_center.y) - cam.cy) / cam.fy;
//
//    // ====================================================
//    //  Apply the radial and tangential distortion model [Double calculation]
//    // ====================================================
//
//    // 3.2. Compute radial distance squared
//    double r_sq = xn * xn + yn * yn;
//
//    // 3.3. Compute radial distortion factor
//    double radial_factor = 1.0 + cam.k1 * r_sq + cam.k2 * r_sq * r_sq + cam.k3 * r_sq * r_sq * r_sq;
//
//    // 3.4. Compute radial and tangential distortion components
//    double x_radial = xn * radial_factor;
//    double y_radial = yn * radial_factor;
//
//    double x_tan = 2.0 * cam.p1 * xn * yn + cam.p2 * (r_sq + 2.0 * xn * xn);
//    double y_tan = cam.p1 * (r_sq + 2.0 * yn * yn) + 2.0 * cam.p2 * xn * yn;
//
//    // 3.5. Combine to get the distorted normalized coordinates (xd, yd)
//    double xd = x_radial + x_tan;
//    double yd = y_radial + y_tan;
//
//    // ====================================================
//    //  Distorted normalized coordinates -> Distorted pixel coordinates [convert back to float output]
//    // ====================================================
//    float u_distorted = /*static_cast<float>*/(xd * cam.fx + cam.cx);
//    float v_distorted = /*static_cast<float>*/(yd * cam.fy + cam.cy);
//
//    return make_float2(u_distorted, v_distorted);
//}
//inline __device__ float subsample_bilinear(const float* img, const float2& u_dist, const int width, const int height)
//{
//    // Left top integer coordinates
//    int x0 = __float2int_rd(u_dist.x);
//    int y0 = __float2int_rd(u_dist.y);
//    int x1 = x0 + 1;
//    int y1 = y0 + 1;
//    // boundary checks
//    if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height) return .0f;
//    // Get values from the image
//    float I00 = img[y0 * width + x0];
//    float I10 = img[y0 * width + x1];
//    float I01 = img[y1 * width + x0];
//    float I11 = img[y1 * width + x1];
//    if (I00 == .0f || I10 == .0f || I01 == .0f || I11 == .0f) return .0f;
//    // Compute dx and dy
//    float dx = u_dist.x - static_cast<float>(x0);
//    float dy = u_dist.y - static_cast<float>(y0);
//    // Interpolate along x (horizontal)
//    float I0 = I00 * (1.0f - dx) + I10 * dx; // Top interpolation
//    float I1 = I01 * (1.0f - dx) + I11 * dx; // Bottom interpolation
//    // Final interpolation along y (vertical)
//    return I0 * (1.0f - dy) + I1 * dy;
//}
//
//__device__ bool computeGeoResidualAndJacobianPixel(
//    int x, int y, int idx,
//    const float* depth1, const float3* norm1, const float* depth2,
//    cuCam cam, const double2* raymap, cuRelPose pose,
//    float& e_geo, float Jt_geo[6]
//) {
//    float Z1 = depth1[idx] * 1e-3;
//    if (Z1 <= 0) return false;
//
//    float3 n = norm1[idx];
//
//    float3 P1; // 重建源帧中像素点的3D坐标 p1
//    P1.x = raymap[idx].x * Z1;
//    P1.y = raymap[idx].y * Z1;
//    P1.z = Z1;
//
//    float3 P2; // 将点变换到目标帧坐标系 p2 = T_inv * P1 = R_inv * p1 + t_inv
//    P2.x = pose.R_inv[0] * P1.x + pose.R_inv[1] * P1.y + pose.R_inv[2] * P1.z + pose.t_inv[0];
//    P2.y = pose.R_inv[3] * P1.x + pose.R_inv[4] * P1.y + pose.R_inv[5] * P1.z + pose.t_inv[1];
//    P2.z = pose.R_inv[6] * P1.x + pose.R_inv[7] * P1.y + pose.R_inv[8] * P1.z + pose.t_inv[2];
//
//    float2 u_ideal, u_dist;
//    u_ideal.x = (P2.x * cam.fx) / P2.z + cam.cx;
//    u_ideal.y = (P2.y * cam.fy) / P2.z + cam.cy;
//    u_dist = reproject_ideal_to_distorted(u_ideal, cam);
//    float Z2 = subsample_bilinear(depth2, u_dist, cam.width, cam.height) * 1e-3;
//    //float Z2 = P2.z;
//    if (Z2 == .0f) return false;
//
//    if (abs(Z2 - P2.z) > 5e-3f) return false; // occlusion between objects
//
//    float3 Q2; // 重建目标帧对应点 q
//    Q2.x = (u_ideal.x - cam.cx) * Z2 / cam.fx;
//    Q2.y = (u_ideal.y - cam.cy) * Z2 / cam.fy;
//    Q2.z = Z2;
//
//    float3 Q1;
//    Q1.x = pose.R[0] * Q2.x + pose.R[1] * Q2.y + pose.R[2] * Q2.z + pose.t[0];
//    Q1.y = pose.R[3] * Q2.x + pose.R[4] * Q2.y + pose.R[5] * Q2.z + pose.t[1];
//    Q1.z = pose.R[6] * Q2.x + pose.R[7] * Q2.y + pose.R[8] * Q2.z + pose.t[2];
//
//    // 计算误差 e = n^T * (p1 - q1), E = e^2
//    float ex = P1.x - Q1.x;
//    float ey = P1.y - Q1.y;
//    float ez = P1.z - Q1.z;
//    e_geo = (n.x * ex + n.y * ey + n.z * ez);
//    //if (abs(e) > 3.0f) return;
//
//    // 计算误差对位姿参数的Jacobian（6维：平移vx,vy,vz; 旋转ωx,ωy,ωz）
//    // 平移部分: ∂e/∂v = -n^T
//    Jt_geo[0] = -n.x;  // ∂e/∂vx
//    Jt_geo[1] = -n.y;  // ∂e/∂vy
//    Jt_geo[2] = -n.z;  // ∂e/∂vz
//    // 旋转部分: ∂e/∂ω ≈ -n^T * -[q1]_x (叉积矩阵)
//    // 等价于 Jt[3..5] = (n × q1)
//    Jt_geo[3] = (n.y * Q1.z - n.z * Q1.y); // ∂e/∂ωx
//    Jt_geo[4] = (n.z * Q1.x - n.x * Q1.z); // ∂e/∂ωy
//    Jt_geo[5] = (n.x * Q1.y - n.y * Q1.x); // ∂e/∂ωz
//
//    return true;
//}
//
//__global__ void computeGradAndDiagGlobal(
//    const cuEdge* __restrict__ edges,
//    int numEdges,
//    const cuCam cam,
//    const double2* raymap,
//    float* __restrict__ JTr,          // size = 6 * numFrames, 初始为 0
//    float* __restrict__ Hdiag,        // size = 6 * numFrames, 初始为 0
//    float* __restrict__ d_errormap)   // size = numEdges * width * height
//{
//    // 每个 block 处理一个 edge，每个 thread 处理若干像素
//    int eIdx = blockIdx.z;
//    if (eIdx >= numEdges) return;
//
//    const cuEdge& edge = edges[eIdx];
//
//    int i = edge.i;
//    int j = edge.j;
//
//    int width = cam.width;
//    int height = cam.height;
//
//    int u = blockIdx.x * blockDim.x + threadIdx.x;
//    int v = blockIdx.y * blockDim.y + threadIdx.y;
//    if (u >= width || v >= height) return;
//    int idx = v * width + u;
//    int errIndex = eIdx * width * height + idx;
//
//    float e_geo;
//    float J_rel[6];
//
//    bool ok = computeGeoResidualAndJacobianPixel(
//        u, v, idx,
//        edge.depth_i, edge.normal_i, edge.depth_j,
//        cam, raymap, edge.T_ij,
//        e_geo, J_rel);
//    if (!ok) return;
//
//    d_errormap[errIndex] = e_geo;
//    float w = 1.0f; // 几何权重，可扩展
//
//    // --- J_rel 是 1x6，我们要算 J_i 和 J_j --- //
//
//    // 取出 AdInv_i
//    const float* A = edge.d_AdInv.data; // 6x6 row-major // per-frame Ad(Twc_i^{-1})
//
//    float Ji[6], Jj[6];
//#pragma unroll
//    for (int r = 0; r < 6; ++r) {
//        float tmp = 0.0f;
//#pragma unroll
//        for (int c = 0; c < 6; ++c) {
//            tmp += J_rel[c] * A[r * 6 + c]; // J_rel (1x6) * A(6x6) = 1x6
//        }
//        // Jj =  J_rel * AdInv_i
//        Jj[r] = tmp;
//        // Ji = -J_rel * AdInv_i
//        Ji[r] = -tmp;
//    }
//
//    // 全局索引
//    int base_i = 6 * i;
//    int base_j = 6 * j;
//
//    // g = Σ (-w e J)
//    for (int k = 0; k < 6; ++k) {
//        atomicAdd(&JTr[base_i + k], -w * e_geo * Ji[k]);
//        atomicAdd(&JTr[base_j + k], -w * e_geo * Jj[k]);
//
//        atomicAdd(&Hdiag[base_i + k], w * Ji[k] * Ji[k]);
//        atomicAdd(&Hdiag[base_j + k], w * Jj[k] * Jj[k]);
//    }
//}
//__global__ void applyH_global(
//    const cuEdge* __restrict__ edges,
//    int numEdges,
//    const cuCam cam,
//    const double2* raymap,
//    const float* __restrict__ x,  // size = 6 * numFrames
//    float* __restrict__ y)        // size = 6 * numFrames, 调用前需置零
//{
//    int eIdx = blockIdx.z;
//    if (eIdx >= numEdges) return;
//
//    const cuEdge& edge = edges[eIdx];
//
//    int i = edge.i;
//    int j = edge.j;
//
//    int width = cam.width;
//    int height = cam.height;
//
//    int u = blockIdx.x * blockDim.x + threadIdx.x;
//    int v = blockIdx.y * blockDim.y + threadIdx.y;
//    if (u >= width || v >= height) return;
//    int idx = v * width + u;
//
//    float e_geo;
//    float J_rel[6];
//
//    bool ok = computeGeoResidualAndJacobianPixel(
//        u, v, idx,
//        edge.depth_i, edge.normal_i, edge.depth_j,
//        cam, raymap, edge.T_ij,
//        e_geo, J_rel);
//    if (!ok) return; 
//    
//    float w = 1.0f;
//
//    // 构造 J_i, J_j
//    const float* A = edge.d_AdInv.data;
//
//    float Ji[6], Jj[6];
//#pragma unroll
//    for (int r = 0; r < 6; ++r) {
//        float tmp = 0.0f;
//#pragma unroll
//        for (int c = 0; c < 6; ++c) {
//            tmp += J_rel[c] * A[r * 6 + c];
//        }
//        Jj[r] = tmp;
//        Ji[r] = -tmp;
//    }
//
//    int base_i = 6 * i;
//    int base_j = 6 * j;
//
//    // 从 x 中取对应节点块
//    float s = 0.0f;
//#pragma unroll
//    for (int k = 0; k < 6; ++k) {
//        s += Ji[k] * x[base_i + k] + Jj[k] * x[base_j + k];
//    }
//    s *= w;
//
//    // y_i += J_i^T * s
//    // y_j += J_j^T * s
//#pragma unroll
//    for (int k = 0; k < 6; ++k) {
//        atomicAdd(&y[base_i + k], Ji[k] * s);
//        atomicAdd(&y[base_j + k], Jj[k] * s);
//    }
//}
//
//__global__ void zeroVector(float* y, int n) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < n) y[idx] = 0.0f;
//}
//__global__ void axpy(float* y, const float* x, float alpha, int n) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < n) y[idx] += alpha * x[idx];
//}
//__global__ void applyPreconditionerSingle(
//    const float* r, const float* Minv_diag, float* z, int n)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < n) {
//        z[idx] = Minv_diag[idx] * r[idx];
//    }
//}
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
//__global__ void buildJacobiPreconditioner(
//    const float* __restrict__ Hdiag,
//    float* __restrict__ Minv_diag,
//    int n)
//{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= n) return;
//
//    // 固定第 0 帧：它的 6 个自由度不参与优化 => Minv = 0
//    if (i < 6) {
//        Minv_diag[i] = 0.0f;
//        return;
//    }
//
//    float d = Hdiag[i];
//    if (fabsf(d) > 1e-8f) Minv_diag[i] = 1.0f / d;
//    else Minv_diag[i] = 1.0f;
//}
//__global__ void zeroFirstFrame6(float* x)
//{
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (idx < 6) x[idx] = 0.0f;
//}
//
//__global__ void unitFirstFrame6(float* x)
//{
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (idx < 6) x[idx] = 1.0f;
//}
//
//void pcgSolveDenseBA(
//    const DenseBAData& data,
//    float* d_delta,   // size = 6 * numFrames
//    float* d_b,       // size = 6 * numFrames, 通常就是 d_JTr
//    int maxIters, float tol)
//{
//    const int n = 6 * data.numFrames;
//    const int offset = 6;      // 前 6 个为第 0 帧
//    const int nVar = n - 6;  // 实际参与优化的变量数
//
//    dim3 block1d(64);
//    dim3 grid1d((n + block1d.x - 1) / block1d.x);
//
//    float* d_r = nullptr;
//    float* d_z = nullptr;
//    float* d_p = nullptr;
//    float* d_Ap = nullptr;
//    cudaMalloc(&d_r, n * sizeof(float));
//    cudaMalloc(&d_z, n * sizeof(float));
//    cudaMalloc(&d_p, n * sizeof(float));
//    cudaMalloc(&d_Ap, n * sizeof(float));
//
//    // x = 0
//    zeroVector << <grid1d, block1d >> > (d_delta, n);
//
//    // r = b - A x = b
//    cudaMemcpy(d_r, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice);
//
//    // 确保第 0 帧从一开始就是固定的
//    zeroFirstFrame6 << <1, 6 >> > (d_r);
//    zeroFirstFrame6 << <1, 6 >> > (d_delta);
//
//    // z = M^{-1} r
//    applyPreconditionerSingle << <grid1d, block1d >> > (
//        d_r, data.d_Minv_diag, d_z, n);
//    zeroFirstFrame6 << <1, 6 >> > (d_z);
//
//    // 注意：dot product 只从 offset 开始算
//    thrust::device_ptr<float> r_ptr(d_r + offset);
//    thrust::device_ptr<float> z_ptr(d_z + offset);
//    thrust::device_ptr<float> p_ptr(d_p + offset);
//    thrust::device_ptr<float> Ap_ptr(d_Ap + offset);
//
//    // p = z
//    cudaMemcpy(d_p, d_z, n * sizeof(float), cudaMemcpyDeviceToDevice);
//    zeroFirstFrame6 << <1, 6 >> > (d_p);
//
//    float rz = thrust::inner_product(
//        r_ptr, r_ptr + nVar,
//        z_ptr, 0.0f);
//
//    // 图像分辨率，用来配置 applyH 的网格
//    dim3 block2d(16, 16);
//    dim3 grid2d(
//        (data.cam.width + block2d.x - 1) / block2d.x,
//        (data.cam.height + block2d.y - 1) / block2d.y,
//        data.numEdges);
//
//    for (int iter = 0; iter < maxIters; ++iter)
//    {
//        // Ap = A * p
//        zeroVector << <grid1d, block1d >> > (d_Ap, n);
//        applyH_global << <grid2d, block2d >> > (
//            data.d_edges,
//            data.numEdges,
//            data.cam,
//            data.raymap,
//            d_p,
//            d_Ap);
//
//        zeroFirstFrame6 << <1, 6 >> > (d_Ap);
//
//        float pAp = thrust::inner_product(
//            p_ptr, p_ptr + nVar,
//            Ap_ptr, 0.0f);
//        if (fabsf(pAp) < 1e-20f) break;
//
//        float alpha = rz / pAp;
//
//        // x = x + alpha * p
//        axpy << <grid1d, block1d >> > (d_delta, d_p, alpha, n);
//        zeroFirstFrame6 << <1, 6 >> > (d_delta);  // 再保险一次
//
//        // r = r - alpha * Ap
//        axpy << <grid1d, block1d >> > (d_r, d_Ap, -alpha, n);
//        zeroFirstFrame6 << <1, 6 >> > (d_r);
//
//        // 收敛判断 ||r||
//        float r2 = thrust::inner_product(
//            r_ptr, r_ptr + nVar,
//            r_ptr, 0.0f);
//        float rnorm = sqrtf(r2);
//
//        printf("[PCG] iter %d, rnorm = %e\n", iter, rnorm);
//        if (rnorm < tol) break;
//
//        // z_new = M^{-1} r
//        applyPreconditionerSingle << <grid1d, block1d >> > (
//            d_r, data.d_Minv_diag, d_z, n);
//        zeroFirstFrame6 << <1, 6 >> > (d_z);
//
//        float rz_new = thrust::inner_product(
//            r_ptr, r_ptr + nVar,
//            z_ptr, 0.0f);
//        float beta = rz_new / rz;
//        rz = rz_new;
//
//        // p = beta * p + z
//        scaleAndAdd << <grid1d, block1d >> > (d_p, d_z, beta, n);
//        zeroFirstFrame6 << <1, 6 >> > (d_p);
//    }
//
//    cudaFree(d_r);
//    cudaFree(d_z);
//    cudaFree(d_p);
//    cudaFree(d_Ap);
//}
//
//void showErrorMaps(
//    float* d_errormap,
//    int numEdges,
//    int width,
//    int height)
//{
//    int imgSize = width * height;
//    int totalSize = numEdges * imgSize;
//
//    // 1. 拷贝到 CPU
//    std::vector<float> h_errormap(totalSize);
//    cudaMemcpy(h_errormap.data(),
//        d_errormap,
//        totalSize * sizeof(float),
//        cudaMemcpyDeviceToHost);
//
//    // 2. 为了可视化，一般做一下归一化 & 取绝对值（误差可能有正负）
//    for (int eIdx = 0; eIdx < numEdges; ++eIdx) {
//        float* src = h_errormap.data() + eIdx * imgSize;
//
//        // 用 src 构造一个 CV_32F 的 Mat
//        cv::Mat err32f(height, width, CV_32F, src);  // 不拷贝数据，直接用指针
//        cv::Mat err32f_clone = err32f.clone();       // 如需独立 Mat，可以 clone 一份
//
//        // 取绝对值，看误差大小
//        cv::Mat err_abs;
//        cv::absdiff(err32f_clone, cv::Scalar::all(0), err_abs);
//
//        // 归一化到 [0, 255]
//        cv::Mat err_norm;
//        double minVal, maxVal;
//        cv::minMaxLoc(err_abs, &minVal, &maxVal);
//
//        if (maxVal > minVal) {
//            err_abs.convertTo(err_norm, CV_8U, 255.0 / (maxVal - minVal),
//                -minVal * 255.0 / (maxVal - minVal));
//        }
//        else {
//            // 全零或常数图，直接转 8U
//            err_abs.convertTo(err_norm, CV_8U);
//        }
//
//        // 可选：上色，便于观察
//        cv::Mat err_color;
//        cv::applyColorMap(err_norm, err_color, cv::COLORMAP_JET);
//
//        // 显示
//        std::string winName = "errormap edge " + std::to_string(eIdx);
//        cv::imshow(winName, err_color);
//    }
//
//    // 等待按键（0 表示阻塞），可以换成 1/10ms 做一个动画式刷新
//    cv::waitKey(1);
//}
//using namespace std;
//extern "C" void denseOptPoseSE3PCG(
//    cv::Mat mK, cv::Mat mDistCoeffs, cv::Mat Raymap,
//    vector<ORB_SLAM3::KeyFrame*> vpKFs, vector<pair<ORB_SLAM3::KeyFrame*, Eigen::Matrix4f>>&mvTwc_KF_opt,
//    int maxIter = 20, float tol = 1e-6)
//{
//    // -------- Data filtering --------
//    cout << "[Dense BA] Filtering keyframe data..." << endl;
//    sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);
//    auto it = vpKFs.begin();
//    while (it != vpKFs.end())
//    {
//        ORB_SLAM3::KeyFrame* pKF = *it;
//        if (pKF->isBad() || !pKF->hasValidMSTransform || pKF->depth_KF.empty() || pKF->color_KF.empty())
//        {
//            it = vpKFs.erase(it);  // erase 返回下一个有效迭代器
//            continue;
//        }
//        std::vector<ORB_SLAM3::KeyFrame*> vCov = pKF->GetBestCovisibilityKeyFrames(1); // 取最好的共视关键帧
//        if (vCov.empty() || vCov[0]->isBad() || !vCov[0]->hasValidMSTransform || vCov[0]->depth_KF.empty() || vCov[0]->color_KF.empty())
//        {
//            it = vpKFs.erase(it); // 没有共视帧，或者共视帧为空/坏帧，也删掉当前 pKF
//            continue;
//        }
//        ++it; // 正常情况，继续下一个
//    }
//    cout << "[Dense BA] " << vpKFs.size() << " keyframes would be considered." << endl;
//
//    // -------- Data transfer --------
//    cout << "[Dense BA] Copying data from host to device..." << endl;
//    DenseBAData dbData;
//    // Intrinsic stuff
//    int width = Raymap.cols, height = Raymap.rows;
//    size_t imgSize = width * height;
//    cuCam h_cam;
//    h_cam.width = width; h_cam.height = height;
//    h_cam.fx = mK.at<double>(0, 0); h_cam.fy = mK.at<double>(1, 1);
//    h_cam.cx = mK.at<double>(0, 2); h_cam.cy = mK.at<double>(1, 2);
//    h_cam.k1 = static_cast<double>(mDistCoeffs.at<double>(0));
//    h_cam.k2 = static_cast<double>(mDistCoeffs.at<double>(1));
//    h_cam.p1 = static_cast<double>(mDistCoeffs.at<double>(2));
//    h_cam.p2 = static_cast<double>(mDistCoeffs.at<double>(3));
//    h_cam.k3 = static_cast<double>(mDistCoeffs.at<double>(4));
//    double2* d_raymap;
//    cudaMalloc(&d_raymap, imgSize * sizeof(double2));
//    cudaMemcpy(d_raymap, Raymap.ptr<double>(), imgSize * sizeof(double2), cudaMemcpyHostToDevice);
//    // Map{depth/normal/texture} stuff
//    const int numFrames_important = vpKFs.size();
//    vector<float*>  d_depth(numFrames_important); // ptrs on GPU
//    vector<float3*> d_normal(numFrames_important);// ptrs on GPU
//    for (int i = 0; i < numFrames_important; ++i)
//    {
//        ORB_SLAM3::KeyFrame* pKF = vpKFs[i];
//        //auto Tcw = pKF->GetPose();
//        //Tcw.translation() = Tcw.translation() * 1.1;
//        //pKF->SetPose(Tcw);
//
//        // depth_KF: CV_32FC1
//        cudaMalloc(&d_depth[i], imgSize * sizeof(float));
//        cudaMemcpy(d_depth[i], pKF->depth_KF.ptr<float>(),
//            imgSize * sizeof(float),
//            cudaMemcpyHostToDevice);
//    
//        // normal_KF: CV_32FC3，相当于 float[3]
//        cudaMalloc(&d_normal[i], imgSize * sizeof(float3));
//        cudaMemcpy(d_normal[i], pKF->normal_KF.ptr<float>(),
//            imgSize * sizeof(float3),
//            cudaMemcpyHostToDevice);
//    }
//    cout << "[Dense BA] Constructing edges..." << endl;
//    // Edges
//    vector<cuEdge> d_edges;
//    for (int i = 0; i < numFrames_important; ++i)
//    {
//        /*int i = k;
//        int j = k+1;
//        if (j == i || j >= numFrames_important) j = 0;*/
//        //int j = (i + 1) == numFrames_important ? 0 : i + 1; // simply construct a best covisibility edge first, construct full edge later.
//        ORB_SLAM3::KeyFrame* pKF = vpKFs[i];
//        //ORB_SLAM3::KeyFrame* cov = vpKFs[j];
//        vector<ORB_SLAM3::KeyFrame*> vCov = pKF->GetBestCovisibilityKeyFrames(5); 
//        for (auto cov : vCov) 
//        {
//            auto it = std::find(vpKFs.begin(), vpKFs.end(), cov);
//            std::size_t j = std::distance(vpKFs.begin(), it);
//
//            cuEdge d_egde;
//
//            d_egde.i = i;
//            d_egde.j = j;
//            d_egde.depth_i = d_depth[i];
//            d_egde.depth_j = d_depth[j];
//            d_egde.normal_i = d_normal[i];
//
//            cuRelPose pose;
//            Eigen::Matrix4f Twc_i, Twc_j, Twc_i_inv, initPose;
//            Twc_i = pKF->GetPoseInverse().matrix();
//            Twc_j = cov->GetPoseInverse().matrix();
//            Twc_i_inv = Twc_i.inverse();
//            initPose = Twc_i_inv * Twc_j;
//            Eigen::Matrix3f R, R_inv, ad_R;
//            Eigen::Vector3f t, t_inv, ad_t;
//            R = initPose.block<3, 3>(0, 0); R_inv = R.transpose();
//            t = initPose.block<3, 1>(0, 3); t_inv = -R_inv * t;
//            //if (t.norm() < 1e-4) { cout << "----------------" << endl; continue; } // WTF
//            for (int i = 0; i < 3; ++i) {
//                for (int j = 0; j < 3; ++j) {
//                    pose.R[i * 3 + j] = R(i, j);
//                    pose.R_inv[i * 3 + j] = R_inv(i, j);
//                }
//                pose.t[i] = t[i];
//                pose.t_inv[i] = t_inv[i];
//            }
//            d_egde.T_ij = pose;
//
//            cuAdj AdInv;
//            Eigen::Matrix<float, 6, 6> AdT_inv;
//            ad_R = Twc_i_inv.block<3, 3>(0, 0);
//            ad_t = Twc_i_inv.block<3, 1>(0, 3);
//            AdT_inv.block<3, 3>(0, 0) = ad_R;
//            AdT_inv.block<3, 3>(0, 3) = skew_symmetric(ad_t) * ad_R;
//            AdT_inv.block<3, 3>(3, 0) = Eigen::Matrix3f::Zero();
//            AdT_inv.block<3, 3>(3, 3) = ad_R;
//            std::memcpy(AdInv.data, AdT_inv.data(), 36 * sizeof(float));
//            d_egde.d_AdInv = AdInv;
//
//            d_edges.push_back(d_egde);
//            cout << "creating Edge with keyframe i: " << i << " and keyframe j: " << j << endl;
//        }
//    }
//
//    dbData.cam = h_cam;
//    dbData.raymap = d_raymap;
//    dbData.numFrames = static_cast<int>(numFrames_important);
//    dbData.numEdges = static_cast<int>(d_edges.size());
//
//    cuEdge* d_edges_dev = nullptr;
//    cudaMalloc(&d_edges_dev, dbData.numEdges * sizeof(cuEdge));
//    cudaMemcpy(d_edges_dev, d_edges.data(),
//        dbData.numEdges * sizeof(cuEdge),
//        cudaMemcpyHostToDevice);
//    dbData.d_edges = d_edges_dev;
//
//    cout << "[Dense BA] Data copied!" << endl;
//
//    // 分配 PCG 需要的向量：JTr、Hdiag、Minv 和 δξ
//    const int n = 6 * dbData.numFrames;
//
//    float* d_JTr = nullptr;
//    float* d_Hdiag = nullptr;
//    float* d_delta = nullptr;
//    float* d_Minv = nullptr;
//
//    cudaMalloc(&d_JTr, n * sizeof(float));
//    cudaMalloc(&d_Hdiag, n * sizeof(float));
//    cudaMalloc(&d_delta, n * sizeof(float));
//    cudaMalloc(&d_Minv, n * sizeof(float));
//
//    dbData.d_Minv_diag = d_Minv;
// 
//    // debug
//    size_t totalSize = dbData.numEdges * imgSize;
//    float* d_errormap;
//    cudaMalloc(&d_errormap, totalSize * sizeof(float));
//
//    // 1) 计算 JTr 和 diag(H)
//    dim3 block2d(16, 16);
//    dim3 grid2d(
//        (h_cam.width + block2d.x - 1) / block2d.x,
//        (h_cam.height + block2d.y - 1) / block2d.y,
//        dbData.numEdges);
//
//    dim3 block1d(64);
//    dim3 grid1d((n + block1d.x - 1) / block1d.x);
//    
//    vector<Eigen::Matrix4f> vTwc_opt(dbData.numFrames);
//
//    for (int iter = 0; iter < maxIter; ++iter) 
//    {
//        // 每次迭代需更新：DenseBAData->d_Minv_diag, ->cuEdge{cuRelPose{R,t,R_inv,t_inv}, cuAdj}
//        // 清零 JTr 和 Hdiag
//        zeroVector << <grid1d, block1d >> > (d_JTr, n);
//        zeroVector << <grid1d, block1d >> > (d_Hdiag, n);
//
//        cudaMemset(d_errormap, 0, totalSize * sizeof(float)); // debug
//
//        computeGradAndDiagGlobal << <grid2d, block2d >> > (
//            dbData.d_edges,
//            dbData.numEdges,
//            dbData.cam,
//            dbData.raymap,
//            d_JTr,
//            d_Hdiag,
//            d_errormap
//            );
//
//        cudaError_t cudaStatus;
//        cudaStatus = cudaGetLastError(); // Check for any errors launching the kernel
//        if (cudaStatus != cudaSuccess) {
//            fprintf(stderr, "computeErrorAndJacobianCU launch failed: %s\n", cudaGetErrorString(cudaStatus));
//            system("pause");
//        }
//        cudaStatus = cudaDeviceSynchronize();
//        if (cudaStatus != cudaSuccess) {
//            fprintf(stderr, "cudaDeviceSynchronize launch failed: %s\n", cudaGetErrorString(cudaStatus));
//            system("pause");
//        }
//
//        zeroFirstFrame6 << <1, 6 >> > (d_JTr);
//        zeroFirstFrame6 << <1, 6 >> > (d_Hdiag);
//
//        // debug
//        showErrorMaps(d_errormap, dbData.numEdges, h_cam.width, h_cam.height);
//        vector<float> h_JTr(n), h_Hdiag(n), h_Minv(n);
//        cudaMemcpy(h_JTr.data(), d_JTr,
//            n * sizeof(float), cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_Hdiag.data(), d_Hdiag,
//            n * sizeof(float), cudaMemcpyDeviceToHost);
//
//        cout << "JTr: ";
//        for (int i = 0; i < n; i++) {
//            cout << h_JTr[i] << "\t";
//            if (i == 5) cout << endl;
//        }
//        cout << endl;
//        /*for (int i = 0; i < n; i++)
//            cout << h_Hdiag[i] << "\t";
//        cout << endl;*/
//
//        // 2) 构造 Jacobi 预条件器 Minv = diag(H)^{-1}
//        buildJacobiPreconditioner << <grid1d, block1d >> > (
//            d_Hdiag, d_Minv, n);
//
//        /*cudaMemcpy(h_Minv.data(), dbData.d_Minv_diag,
//            n * sizeof(float), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < n; i++)
//            cout << h_Minv[i] << "\t";*/
//
//            // 3) 用 PCG 解 H δξ = -JTr
//            // 注意：你的 computeGradAndDiagGlobal 里 g = Σ (-w e J)，
//            // 这里 d_JTr 就是「-J^T r」，所以方程是 H δ = d_JTr
//            // 下面假定 d_JTr 就是方程右边 b：
//        pcgSolveDenseBA(dbData, d_delta, d_JTr,
//            /*maxItersPCG*/10, 1e-8f);
//        cudaStatus = cudaGetLastError(); // Check for any errors launching the kernel
//        if (cudaStatus != cudaSuccess) {
//            fprintf(stderr, "computeErrorAndJacobianCU launch failed: %s\n", cudaGetErrorString(cudaStatus));
//            system("pause");
//        }
//
//        // 4) 拷回 δξ，并更新每个 KF 的 Twc
//        std::vector<float> h_delta(n);
//        cudaMemcpy(h_delta.data(), d_delta,
//            n * sizeof(float), cudaMemcpyDeviceToHost);
//
//        for (int i = 0; i < dbData.numFrames; ++i)
//        {
//            //// 固定第 0 帧：不更新
//            //if (i == 0) {
//            //    Eigen::Matrix4f Twc_0 = vpKFs[0]->GetPoseInverse().matrix();
//            //    vTwc_opt[0] = Twc_0;
//            //    continue;
//            //}
//
//            // 取出该帧的 6 维增量
//            float vx = h_delta[6 * i + 0];
//            float vy = h_delta[6 * i + 1];
//            float vz = h_delta[6 * i + 2];
//            float wx = h_delta[6 * i + 3];
//            float wy = h_delta[6 * i + 4];
//            float wz = h_delta[6 * i + 5];
//            cout << "frame " << i << " delta: " << vx << " " << vy << " " << vz << " " << wx << " " << wy << " " << wz << endl;
//
//            // === 4) 和原来一样，用 δ 更新 R, t ===
//            Eigen::Vector3f v(vx, vy, vz);
//            Eigen::Vector3f w(wx, wy, wz);
//
//            float theta = w.norm();
//            Eigen::Matrix3f R_inc = Eigen::Matrix3f::Identity();
//            Eigen::Vector3f t_inc = v;
//
//            if (theta > 1e-12f) {
//                Eigen::Vector3f axis = w / theta;
//                Eigen::AngleAxisf aa(theta, axis);
//                R_inc = aa.toRotationMatrix();
//            }
//
//            ORB_SLAM3::KeyFrame* pKF = vpKFs[i];
//            // 当前 Twc_i
//            Eigen::Matrix4f Twc_i = pKF->GetPoseInverse().matrix();
//
//            // 左乘增量：Twc_i' = T_inc * Twc_i
//            Eigen::Matrix4f T_inc = Eigen::Matrix4f::Identity();
//            T_inc.block<3, 3>(0, 0) = R_inc;
//            T_inc.block<3, 1>(0, 3) = t_inc;
//
//            Eigen::Matrix4f Twc_new = T_inc * Twc_i;
//            vTwc_opt[i] = Twc_new;
//
//            cout << "INIT: " << Twc_i.matrix() << endl;
//            cout << "OPTI: " << Twc_new.matrix() << endl;
//
//            // 把结果写回 KeyFrame，方便后续迭代使用
//            Eigen::Matrix4f Tcw_new = Twc_new.inverse();
//            Sophus::SE3f Tcw_se3(Tcw_new.block<3, 3>(0, 0),
//                Tcw_new.block<3, 1>(0, 3));
//            pKF->SetPose(Tcw_se3);
//        }
//
//        for (int j = 0; j < dbData.numEdges; ++j)
//        {
//            cuEdge& edge = d_edges[j]; // 沿用vector<cuEdge> d_edges.
//            cuRelPose pose;
//            Eigen::Matrix4f Twc_i, Twc_j, Twc_i_inv, initPose;
//            Twc_i = vpKFs[edge.i]->GetPoseInverse().matrix();
//            Twc_j = vpKFs[edge.j]->GetPoseInverse().matrix();
//            Twc_i_inv = Twc_i.inverse();
//            initPose = Twc_i_inv * Twc_j;
//            Eigen::Matrix3f R, R_inv, ad_R;
//            Eigen::Vector3f t, t_inv, ad_t;
//            R = initPose.block<3, 3>(0, 0); R_inv = R.transpose();
//            t = initPose.block<3, 1>(0, 3); t_inv = -R_inv * t;
//            //if (t.norm() < 1e-4) { cout << "----------------" << endl; continue; } // WTF
//            for (int i = 0; i < 3; ++i) {
//                for (int j = 0; j < 3; ++j) {
//                    pose.R[i * 3 + j] = R(i, j);
//                    pose.R_inv[i * 3 + j] = R_inv(i, j);
//                }
//                pose.t[i] = t[i];
//                pose.t_inv[i] = t_inv[i];
//            }
//            edge.T_ij = pose;
//
//            cuAdj AdInv;
//            Eigen::Matrix<float, 6, 6> AdT_inv;
//            ad_R = Twc_i_inv.block<3, 3>(0, 0);
//            ad_t = Twc_i_inv.block<3, 1>(0, 3);
//            AdT_inv.block<3, 3>(0, 0) = ad_R;
//            AdT_inv.block<3, 3>(0, 3) = skew_symmetric(ad_t) * ad_R;
//            AdT_inv.block<3, 3>(3, 0) = Eigen::Matrix3f::Zero();
//            AdT_inv.block<3, 3>(3, 3) = ad_R;
//            std::memcpy(AdInv.data, AdT_inv.data(), 36 * sizeof(float));
//            edge.d_AdInv = AdInv;
//        }
//        cudaFree(d_edges_dev);
//        d_edges_dev = nullptr;
//        cudaMalloc(&d_edges_dev, dbData.numEdges * sizeof(cuEdge));
//        cudaMemcpy(d_edges_dev, d_edges.data(),
//            dbData.numEdges * sizeof(cuEdge),
//            cudaMemcpyHostToDevice);
//        dbData.d_edges = d_edges_dev;
//    }
//
//    
//    mvTwc_KF_opt.clear();
//    for (int i = 0; i < dbData.numFrames; ++i) {
//        mvTwc_KF_opt.emplace_back(vpKFs[i], vTwc_opt[i]);
//    }
//
//    // 释放资源
//    cudaFree(d_JTr);
//    cudaFree(d_Hdiag);
//    cudaFree(d_delta);
//    cudaFree(d_Minv);
//
//    cudaFree(d_raymap);
//    cudaFree(d_errormap);
//
//    for (int i = 0; i < vpKFs.size(); ++i) {
//        cudaFree(d_depth[i]);
//        cudaFree(d_normal[i]);
//    }
//}
//
