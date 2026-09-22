
// ============================================================================
// PCGsolver_full_optimized.cu
// Drop-in replacement for the original PCGsolver.cu (Dense point-cloud BA backend)
// Major speedups:
//   - One-time (per GN iteration) per-edge normal equation build (A_e 6x6, g_e 6x1)
//   - PCG SpMV uses per-edge blocks (O(numEdges)), NOT per-pixel re-linearization
//   - Block-level reductions reduce global atomics by ~100x-1000x
//   - Block-Jacobi 6x6 preconditioner (fewer PCG iterations)
//   - cuBLAS dot products with DEVICE pointer mode (no thrust, no host scalars)
//
// Notes:
//   - Keeps the exported API: extern "C" void denseOptPoseSE3PCG(...)
//   - Requires your real DenseOptimizer.cuh to define: cuCam, cuRelPose, cuAdj, cuEdge, DenseBAData
//   - Requires ORB_SLAM3::KeyFrame providing: isBad(), hasValidMSTransform, depth_KF, normal_KF,
//     color_KF, GetBestCovisibilityKeyFrames(int), GetPoseInverse(), SetPose(...), KeyFrame::lId comparator
//
// Build:
//   nvcc -O3 --use_fast_math -lineinfo ... -lcublas
// ============================================================================

#include "DenseOptimizer.cuh"
#include "KeyFrame.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstring>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  std::abort(); } } while(0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
  std::abort(); } } while(0)
#endif

// -------------------------------
// Portable read-only load helper
//   - __ldg is only available for certain SM targets.
//   - Some toolchains have limited overload coverage.
//   - LDG() falls back to a normal global load when __ldg isn't available.
// -------------------------------
#ifndef LDG
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
#define LDG(p) __ldg(p)
#else
#define LDG(p) (*(p))
#endif
#endif

// -------------------------------
// Tunables
// -------------------------------
static constexpr int kBlockX = 16;
static constexpr int kBlockY = 16;
static constexpr int kThreadsPerBlock = kBlockX * kBlockY; // 256
static constexpr int kWarpsPerBlock = kThreadsPerBlock / 32; // 8

static constexpr float kDepthScale = 1e-3f;     // mm -> m
static constexpr float kOcclusionThresh = 5e-3f; // 5mm
static constexpr float kPhoGain = 10.0f;        // faster converge
static constexpr float kDamping = 1e-6f;        // for block inversion
static constexpr int   kPCGIters = 10;          // match your old default

static constexpr float kWGeo = 1.0f;
static constexpr float kWPho = 1.0f;

// ============================================================================
// Device helpers
// ============================================================================
__forceinline__ __device__ float warpReduceSum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__forceinline__ __device__ unsigned int warpReduceSumUInt(unsigned int v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// Map (r,c) in symmetric 6x6 to packed upper-triangular index [0,20]
__forceinline__ __device__ int sym6_idx(int r, int c) {
    int a = r, b = c;
    if (b < a) { int t = a; a = b; b = t; }
    // start(a) = a*6 - a*(a-1)/2
    int start = a * 6 - (a * (a - 1)) / 2;
    return start + (b - a);
}

__forceinline__ __device__ float2 reproject_ideal_to_distorted(const float2& p_ideal_center, const cuCam cam)
{
    // Keep double math (same spirit as original) for distortion stability.
    double xn = (double(p_ideal_center.x) - cam.cx) / cam.fx;
    double yn = (double(p_ideal_center.y) - cam.cy) / cam.fy;

    double r2 = xn * xn + yn * yn;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double radial = 1.0 + cam.k1 * r2 + cam.k2 * r4 + cam.k3 * r6;

    double x_radial = xn * radial;
    double y_radial = yn * radial;

    double x_tan = 2.0 * cam.p1 * xn * yn + cam.p2 * (r2 + 2.0 * xn * xn);
    double y_tan = cam.p1 * (r2 + 2.0 * yn * yn) + 2.0 * cam.p2 * xn * yn;

    double xd = x_radial + x_tan;
    double yd = y_radial + y_tan;

    return make_float2(float(xd * cam.fx + cam.cx), float(yd * cam.fy + cam.cy));
}

// Original behavior: if any of 4 samples == 0 => invalid (return 0)
__forceinline__ __device__ float subsample_bilinear_strict0(
    const float* __restrict__ img, const float2& u_dist, int width, int height) // (0, ∞) valid
{
    int x0 = __float2int_rd(u_dist.x);
    int y0 = __float2int_rd(u_dist.y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if ((unsigned)x0 >= (unsigned)width || (unsigned)y0 >= (unsigned)height) return 0.0f;
    if ((unsigned)x1 >= (unsigned)width || (unsigned)y1 >= (unsigned)height) return 0.0f;

    int idx00 = y0 * width + x0;
    int idx10 = y0 * width + x1;
    int idx01 = y1 * width + x0;
    int idx11 = y1 * width + x1;

    float I00 = LDG(img + idx00);
    float I10 = LDG(img + idx10);
    float I01 = LDG(img + idx01);
    float I11 = LDG(img + idx11);

    if (I00 == 0.f || I10 == 0.f || I01 == 0.f || I11 == 0.f) return 0.0f;

    float dx = u_dist.x - float(x0);
    float dy = u_dist.y - float(y0);

    float I0 = fmaf(dx, (I10 - I00), I00);
    float I1 = fmaf(dx, (I11 - I01), I01);
    return fmaf(dy, (I1 - I0), I0);
}

__forceinline__ __device__ float subsample_bilinear_pad(
    const float* __restrict__ img, const float2& u_dist, int width, int height, int pad = 20) // [0, ∞) valid
{
    int x0 = __float2int_rd(u_dist.x);
    int y0 = __float2int_rd(u_dist.y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if ((unsigned)(x0 - pad) >= (unsigned)(width - 2 * pad) || (unsigned)(y0 - pad) >= (unsigned)(height - 2 * pad)) return -1.0f;
    if ((unsigned)(x1 - pad) >= (unsigned)(width - 2 * pad) || (unsigned)(y1 - pad) >= (unsigned)(height - 2 * pad)) return -1.0f;

    int idx00 = y0 * width + x0;
    int idx10 = y0 * width + x1;
    int idx01 = y1 * width + x0;
    int idx11 = y1 * width + x1;

    float I00 = LDG(img + idx00);
    float I10 = LDG(img + idx10);
    float I01 = LDG(img + idx01);
    float I11 = LDG(img + idx11);

    float dx = u_dist.x - float(x0);
    float dy = u_dist.y - float(y0);

    float I0 = fmaf(dx, (I10 - I00), I00);
    float I1 = fmaf(dx, (I11 - I01), I01);
    return fmaf(dy, (I1 - I0), I0);
}

// Compute point-to-plane residual and J_rel (1x6) for this pixel.
// J_rel is Jacobian wrt the *relative* pose perturbation (same as your original J_rel).
__forceinline__ __device__ bool computeGeoResidualAndJrel(
    int idx,
    const float* __restrict__ depth1,
    const float3* __restrict__ norm1,
    const float* __restrict__ depth2,
    const cuCam cam,
    const double2* __restrict__ raymap,
    const cuRelPose& pose, // pose.R,t = T_{i<-j} ; pose.R_inv,t_inv = T_{j<-i}
    float& e_geo,
    float J_rel[6])
{
    float Z1 = LDG(depth1 + idx) * kDepthScale;
    if (Z1 <= 0.f) return false;

    float3 n = norm1[idx];

    // P1 from raymap
    double2 r = raymap[idx];
    float3 P1 = make_float3(float(r.x) * Z1, float(r.y) * Z1, Z1);

    // P2 = T_{j<-i} * P1  (using pose inverse members)
    float3 P2;
    P2.x = pose.R_inv[0] * P1.x + pose.R_inv[1] * P1.y + pose.R_inv[2] * P1.z + pose.t_inv[0];
    P2.y = pose.R_inv[3] * P1.x + pose.R_inv[4] * P1.y + pose.R_inv[5] * P1.z + pose.t_inv[1];
    P2.z = pose.R_inv[6] * P1.x + pose.R_inv[7] * P1.y + pose.R_inv[8] * P1.z + pose.t_inv[2];
    if (P2.z <= 1e-6f) return false;

    // project to ideal pixel
    float2 u_ideal;
    u_ideal.x = (P2.x * cam.fx) / P2.z + cam.cx;
    u_ideal.y = (P2.y * cam.fy) / P2.z + cam.cy;

    // distort + sample depth2
    float2 u_dist = reproject_ideal_to_distorted(u_ideal, cam);
    float Z2 = subsample_bilinear_strict0(depth2, u_dist, cam.width, cam.height) * kDepthScale;
    if (Z2 == 0.f) return false;
    if (fabsf(Z2 - P2.z) > kOcclusionThresh) return false;

    // Q2 in camera j (using ideal coords)
    float3 Q2;
    Q2.x = (u_ideal.x - cam.cx) * Z2 / cam.fx;
    Q2.y = (u_ideal.y - cam.cy) * Z2 / cam.fy;
    Q2.z = Z2;

    // Q1 = T_{i<-j} * Q2  (using pose.R,t)
    float3 Q1;
    Q1.x = pose.R[0] * Q2.x + pose.R[1] * Q2.y + pose.R[2] * Q2.z + pose.t[0];
    Q1.y = pose.R[3] * Q2.x + pose.R[4] * Q2.y + pose.R[5] * Q2.z + pose.t[1];
    Q1.z = pose.R[6] * Q2.x + pose.R[7] * Q2.y + pose.R[8] * Q2.z + pose.t[2];

    // residual e = n^T (P1 - Q1)
    float3 d = make_float3(P1.x - Q1.x, P1.y - Q1.y, P1.z - Q1.z);
    e_geo = fmaf(n.x, d.x, fmaf(n.y, d.y, n.z * d.z));

    // J_rel (same as your original)
    J_rel[0] = -n.x;
    J_rel[1] = -n.y;
    J_rel[2] = -n.z;
    J_rel[3] = (n.y * Q1.z - n.z * Q1.y);
    J_rel[4] = (n.z * Q1.x - n.x * Q1.z);
    J_rel[5] = (n.x * Q1.y - n.y * Q1.x);

    return true;
}

// Compute fiducial photometric residual and J_rel (1x6) for this pixel.
// J_rel is Jacobian wrt the *relative* pose perturbation (same as your original J_rel).
__forceinline__ __device__ bool computePhoResidualAndJrel(
    int idx,
    const float* __restrict__ I_i,
    const float* __restrict__ I_i_gx,
    const float* __restrict__ I_i_gy,
    const float* __restrict__ I_j,
    const float* __restrict__ D_j,
    const uint8_t* __restrict__ mask_j,
    const cuCam cam,
    const double2* __restrict__ raymap,
    const cuRelPose& pose, // pose.R,t = T_{i<-j} ; pose.R_inv,t_inv = T_{j<-i}
    float& e_pho,
    float J_rel[6])
{
    uint8_t M2 = LDG(mask_j + idx);
    if (M2 == 0) return false;

    float Z2 = LDG(D_j + idx) * kDepthScale;
    if (Z2 <= 0.f) return false;

    float I2 = I_j[idx];

    // P2 from raymap
    double2 r = raymap[idx];
    float3 P2 = make_float3(float(r.x) * Z2, float(r.y) * Z2, Z2);

    // P1 = T_{i<-j} * P2  (using pose members)
    float3 P1;
    P1.x = pose.R[0] * P2.x + pose.R[1] * P2.y + pose.R[2] * P2.z + pose.t[0];
    P1.y = pose.R[3] * P2.x + pose.R[4] * P2.y + pose.R[5] * P2.z + pose.t[1];
    P1.z = pose.R[6] * P2.x + pose.R[7] * P2.y + pose.R[8] * P2.z + pose.t[2];
    if (P1.z <= 1e-6f) return false;

    // project to ideal pixel
    float2 u_ideal;
    u_ideal.x = (P1.x * cam.fx) / P1.z + cam.cx;
    u_ideal.y = (P1.y * cam.fy) / P1.z + cam.cy;

    // distort + sample depth2
    float2 u_dist = reproject_ideal_to_distorted(u_ideal, cam);
    float I1 = subsample_bilinear_pad(I_i, u_dist, cam.width, cam.height);
    if (I1 < 0) return false;

    e_pho = (I2 - I1);

    // 计算误差对位姿参数的Jacobian（6维：平移vx,vy,vz; 旋转ωx,ωy,ωz）
    float gx = subsample_bilinear_pad(I_i_gx, u_dist, cam.width, cam.height);
    float gy = subsample_bilinear_pad(I_i_gy, u_dist, cam.width, cam.height);

    // 平移部分: ∂e/∂v = -∇I1(u1) * ∂π/∂p1 * E
    J_rel[0] = -gx * (cam.fx / P1.z);  // ∂e/∂vx
    J_rel[1] = -gy * (cam.fy / P1.z);  // ∂e/∂vy
    J_rel[2] = -(J_rel[0] * P1.x + J_rel[1] * P1.y) / P1.z;  // ∂e/∂vz

    // 旋转部分: ∂e/∂ω ≈ -∇I1(u1) * ∂π/∂p1 * -[p1]_x (叉积矩阵)
    J_rel[3] = P1.y * J_rel[2] + cam.fy * gy; // ∂e/∂ωx
    J_rel[4] = -P1.x * J_rel[2] - cam.fx * gx; // ∂e/∂ωy
    J_rel[5] = -J_rel[0] * P1.y + J_rel[1] * P1.x; // ∂e/∂ωz

    return true;
}

// ============================================================================
// KERNEL 1: Build per-edge normal equations (A_e 6x6 symmetric packed-21, g_e 6)
// grid = (ceil(W/16), ceil(H/16), numEdges), block=(16,16)
// edge_g [E,6], edge_A21 [E,21] must be zeroed before launch.
// ============================================================================
__global__ void buildEdgeNormalEquations(
    const cuEdge* __restrict__ edges,
    int numEdges,
    const cuCam cam,
    const double2* __restrict__ raymap,
    float* __restrict__ edge_g,     // [E,6]
    float* __restrict__ edge_A21,   // [E,21]
    float w_geo,
    float w_pho,
    float k_pho,
    float* __restrict__ geo_sum_sq,       // optional (global) sum of e_geo^2 (UNWEIGHTED)
    unsigned int* __restrict__ geo_count, // optional (global) count of valid geo residuals
    float* __restrict__ pho_sum_sq,       // optional (global) sum of e_pho^2 (UNWEIGHTED)
    unsigned int* __restrict__ pho_count  // optional (global) count of valid pho residuals
)
{
    // IMPORTANT:
    //  - This kernel uses warp-level reductions. Do NOT early-return per-thread,
    //    otherwise __shfl_down_sync will be invoked with inactive lanes (UB -> NaNs).
    int e = (int)blockIdx.z;
    if (e >= numEdges) return; // uniform for the whole block (safe)

    const cuEdge& edge = edges[e];

    int u = (int)blockIdx.x * kBlockX + (int)threadIdx.x;
    int v = (int)blockIdx.y * kBlockY + (int)threadIdx.y;

    bool in = ((unsigned)u < (unsigned)cam.width) && ((unsigned)v < (unsigned)cam.height);

    // Per-thread accumulators (default 0)
    float g[6];
#pragma unroll
    for (int k = 0; k < 6; ++k) g[k] = 0.f;

    float A21[21];
#pragma unroll
    for (int t = 0; t < 21; ++t) A21[t] = 0.f;

    float geo_res2 = 0.f;
    float pho_res2 = 0.f;
    unsigned int geo_cnt = 0u;
    unsigned int pho_cnt = 0u;

    if (in) {
        int idx = v * cam.width + u;

        float e_geo = 0.f;
        float J_geo_rel[6];
        bool geo_ok = computeGeoResidualAndJrel(
            idx, edge.depth_i, edge.normal_i, edge.depth_j,
            cam, raymap, edge.T_ij,
            e_geo, J_geo_rel);

        float e_pho = 0.f;
        float J_pho_rel[6];
        bool pho_ok = computePhoResidualAndJrel(
            idx, edge.I_i, edge.I_i_gx, edge.I_i_gy, edge.I_j, edge.depth_j, edge.mask_j,
            cam, raymap, edge.T_ij,
            e_pho, J_pho_rel);

        for (int i = 0; i < 6; ++i) { // ablation study
            J_geo_rel[i] = 0.0f;
            //J_pho_rel[i] = 0.0f;
        }

        // Map J_rel into the same per-node variable space used by the solver.
        // Jj = J_rel * AdInv_i  (same mapping as the original geo-only path)
        float Jj_geo[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };
        float Jj_pho[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };
        const float* A = edge.d_AdInv.data; // 6x6 row-major

        if (geo_ok) {
#pragma unroll
            for (int r = 0; r < 6; ++r) {
                float tmp = 0.f;
#pragma unroll
                for (int c = 0; c < 6; ++c) tmp = fmaf(J_geo_rel[c], A[r * 6 + c], tmp);
                Jj_geo[r] = tmp;
            }
            geo_res2 = e_geo * e_geo; // UNWEIGHTED for monitoring
            geo_cnt = 1u;
        }
        if (pho_ok) {
#pragma unroll
            for (int r = 0; r < 6; ++r) {
                float tmp = 0.f;
#pragma unroll
                for (int c = 0; c < 6; ++c) tmp = fmaf(J_pho_rel[c], A[r * 6 + c], tmp);
                Jj_pho[r] = tmp;
            }
            pho_res2 = e_pho * e_pho; // UNWEIGHTED for monitoring
            pho_cnt = 1u;
        }

        // Normal equations: accumulate geo + pho (weighted)
        if (geo_ok || pho_ok) {
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                float gg = 0.f;
                if (geo_ok) gg = fmaf(w_geo * e_geo, Jj_geo[k], gg);
                if (pho_ok) gg = fmaf(w_pho * e_pho, Jj_pho[k], gg);
                g[k] = gg;
            }

#pragma unroll
            for (int a = 0; a < 6; ++a) {
#pragma unroll
                for (int b = a; b < 6; ++b) {
                    int id = sym6_idx(a, b);
                    float vv = 0.f;
                    if (geo_ok) vv = fmaf(w_geo * Jj_geo[a], Jj_geo[b], vv);
                    if (pho_ok) vv = fmaf(w_pho * Jj_pho[a], Jj_pho[b], vv);
                    A21[id] = vv;
                }
            }
        }
    }
    // Block-level reduction:
        //   - 29 floats: (6 g + 21 A + 2 residual sumsq: geo + pho)
        //   - 2 uint32 : (geo_count, pho_count)
    int tid = (int)threadIdx.y * kBlockX + (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    __shared__ float shf[kWarpsPerBlock][29];
    __shared__ unsigned int shc_geo[kWarpsPerBlock];
    __shared__ unsigned int shc_pho[kWarpsPerBlock];

    float v0[29];
#pragma unroll
    for (int i0 = 0; i0 < 6; ++i0) v0[i0] = g[i0];
#pragma unroll
    for (int i0 = 0; i0 < 21; ++i0) v0[6 + i0] = A21[i0];
    v0[27] = geo_res2;
    v0[28] = pho_res2;

#pragma unroll
    for (int i0 = 0; i0 < 29; ++i0) {
        float s = warpReduceSum(v0[i0]);
        if (lane == 0) shf[warp][i0] = s;
    }
    {
        unsigned int cg = warpReduceSumUInt(geo_cnt);
        unsigned int cp = warpReduceSumUInt(pho_cnt);
        if (lane == 0) {
            shc_geo[warp] = cg;
            shc_pho[warp] = cp;
        }
    }
    __syncthreads();

    // First warp reduces warp sums
    if (warp == 0) {
#pragma unroll
        for (int i0 = 0; i0 < 29; ++i0) {
            float s = (lane < kWarpsPerBlock) ? shf[lane][i0] : 0.f;
            s = warpReduceSum(s);
            if (lane == 0) shf[0][i0] = s;
        }
        unsigned int cg = (lane < kWarpsPerBlock) ? shc_geo[lane] : 0u;
        unsigned int cp = (lane < kWarpsPerBlock) ? shc_pho[lane] : 0u;
        cg = warpReduceSumUInt(cg);
        cp = warpReduceSumUInt(cp);
        if (lane == 0) {
            shc_geo[0] = cg;
            shc_pho[0] = cp;
        }
    }
    __syncthreads();

    // One atomic per block to the per-edge accumulators
    if (tid == 0) {
        float* eg = edge_g + e * 6;
        float* eA = edge_A21 + e * 21;
#pragma unroll
        for (int k = 0; k < 6; ++k) atomicAdd(eg + k, shf[0][k]);
#pragma unroll
        for (int k = 0; k < 21; ++k) atomicAdd(eA + k, shf[0][6 + k]);

        if (geo_sum_sq) atomicAdd(geo_sum_sq, shf[0][27]);
        if (geo_count) atomicAdd(geo_count, shc_geo[0]);

        if (pho_sum_sq) atomicAdd(pho_sum_sq, shf[0][28]);
        if (pho_count) atomicAdd(pho_count, shc_pho[0]);
    }
}

// ============================================================================
// KERNEL 2: Scatter per-edge A_e,g_e into global b and per-node Hii blocks.
// b_i += g_e, b_j -= g_e
// Hii_i += A_e, Hii_j += A_e
// Gauge: node 0 fixed => skip writing into node 0 slots.
// ============================================================================
__global__ void scatterEdgesToSystem(
    const cuEdge* __restrict__ edges,
    int numEdges,
    const float* __restrict__ edge_g,   // [E,6]
    const float* __restrict__ edge_A21, // [E,21]
    float* __restrict__ b,              // [F,6]
    float* __restrict__ Hii21)          // [F,21]
{
    int e = (int)blockIdx.x;
    if (e >= numEdges) return;

    int lane = (int)threadIdx.x;

    const cuEdge& edge = edges[e];
    int i = edge.i;
    int j = edge.j;

    const float* g = edge_g + e * 6;
    const float* A21 = edge_A21 + e * 21;

    if (lane < 6) {
        float gv = LDG(g + lane);
        if (i != 0) atomicAdd(&b[i * 6 + lane], gv);
        if (j != 0) atomicAdd(&b[j * 6 + lane], -gv);
    }
    if (lane < 21) {
        float av = LDG(A21 + lane);
        if (i != 0) atomicAdd(&Hii21[i * 21 + lane], av);
        if (j != 0) atomicAdd(&Hii21[j * 21 + lane], av);
    }
}

// ============================================================================
// KERNEL 3: Invert per-node 6x6 diagonal blocks (block-Jacobi) using Cholesky.
// Input: Hii21 packed-21; Output: Minv36 row-major 6x6 per node.
// ============================================================================
__global__ void invertBlockJacobi6x6(
    const float* __restrict__ Hii21,
    float* __restrict__ Minv36,
    int numFrames,
    float lambda)
{
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= numFrames) return;

    float* out = Minv36 + i * 36;

    // fixed node
    if (i == 0) {
#pragma unroll
        for (int k = 0; k < 36; ++k) out[k] = 0.f;
        return;
    }

    float A[6][6];
#pragma unroll
    for (int r = 0; r < 6; ++r)
#pragma unroll
        for (int c = 0; c < 6; ++c) A[r][c] = 0.f;

    const float* packed = Hii21 + i * 21;
#pragma unroll
    for (int r = 0; r < 6; ++r) {
#pragma unroll
        for (int c = r; c < 6; ++c) {
            float v = packed[sym6_idx(r, c)];
            A[r][c] = v;
            A[c][r] = v;
        }
    }

    float eps = 1e-9f;
#pragma unroll
    for (int d = 0; d < 6; ++d) A[d][d] += lambda + eps;

    // Cholesky (lower)
    float L[6][6];
#pragma unroll
    for (int r = 0; r < 6; ++r)
#pragma unroll
        for (int c = 0; c < 6; ++c) L[r][c] = 0.f;

#pragma unroll
    for (int r = 0; r < 6; ++r) {
#pragma unroll
        for (int c = 0; c <= r; ++c) {
            float sum = A[r][c];
#pragma unroll
            for (int k = 0; k < c; ++k) sum -= L[r][k] * L[c][k];
            if (r == c) {
                sum = fmaxf(sum, eps);
                L[r][c] = sqrtf(sum);
            }
            else {
                L[r][c] = sum / L[c][c];
            }
        }
    }

    // Invert by solving A x = e_k for k=0..5
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        float y[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };
        y[k] = 1.f;

        // forward: L y = e_k
#pragma unroll
        for (int r = 0; r < 6; ++r) {
            float s = y[r];
#pragma unroll
            for (int c = 0; c < r; ++c) s -= L[r][c] * y[c];
            y[r] = s / L[r][r];
        }

        float x[6] = { 0.f,0.f,0.f,0.f,0.f,0.f };
        // backward: L^T x = y
#pragma unroll
        for (int r = 5; r >= 0; --r) {
            float s = y[r];
#pragma unroll
            for (int c = r + 1; c < 6; ++c) s -= L[c][r] * x[c];
            x[r] = s / L[r][r];
        }

#pragma unroll
        for (int r = 0; r < 6; ++r) out[r * 6 + k] = x[r];
    }
}

// Apply block-Jacobi: z_i = Minv_i * r_i
__global__ void applyBlockJacobi(
    const float* __restrict__ Minv36,
    const float* __restrict__ r,
    float* __restrict__ z,
    int numFrames)
{
    // IMPORTANT:
    //  - Only lanes 0..5 participate. We must NOT use __syncwarp() without a mask,
    //    otherwise lanes 6..31 returning early would make it UB.
    int i = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    if (i >= numFrames) return;

    if (lane >= 6) return; // non-participating lanes

    if (i == 0) { z[lane] = 0.f; return; }

    const unsigned mask = 0x3fu; // lanes 0..5

    float ri = r[i * 6 + lane];

    const float* M = Minv36 + i * 36;

    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < 6; ++c) {
        float rc = __shfl_sync(mask, ri, c);
        acc = fmaf(M[lane * 6 + c], rc, acc);
    }
    z[i * 6 + lane] = acc;
}

// ============================================================================
// KERNEL 4: SpMV y = H x using edge blocks only:
//   y_i += A_e (x_i - x_j)
//   y_j -= A_e (x_i - x_j)
// ============================================================================
__global__ void applyH_fromEdges(
    const cuEdge* __restrict__ edges,
    int numEdges,
    const float* __restrict__ edge_A21,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    // IMPORTANT:
    //  - Only lanes 0..5 participate. Use shuffle with a 0x3f mask; no __syncwarp().
    int e = (int)blockIdx.x;
    if (e >= numEdges) return;

    int lane = (int)threadIdx.x;
    if (lane >= 6) return;

    const cuEdge& edge = edges[e];
    int i = edge.i;
    int j = edge.j;

    const unsigned mask = 0x3fu; // lanes 0..5

    float xi = (i == 0) ? 0.f : LDG(&x[i * 6 + lane]);
    float xj = (j == 0) ? 0.f : LDG(&x[j * 6 + lane]);
    float d_lane = xi - xj; // local d component for this lane

    const float* A21 = edge_A21 + e * 21;

    float t = 0.f;
#pragma unroll
    for (int c = 0; c < 6; ++c) {
        float dc = __shfl_sync(mask, d_lane, c);
        float a = LDG(&A21[sym6_idx(lane, c)]);
        t = fmaf(a, dc, t);
    }

    if (i != 0) atomicAdd(&y[i * 6 + lane], t);
    if (j != 0) atomicAdd(&y[j * 6 + lane], -t);
}

// Simple utilities
__global__ void zeroVector(float* x, int n)
{
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx < n) x[idx] = 0.f;
}

__global__ void pcgUpdateXR(
    float* __restrict__ x,
    float* __restrict__ r,
    const float* __restrict__ p,
    const float* __restrict__ Ap,
    const float* __restrict__ alpha_dev,
    int n)
{
    float a = LDG(alpha_dev);
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= n) return;

    float xi = x[idx] + a * p[idx];
    float ri = r[idx] - a * Ap[idx];

    if (idx < 6) { xi = 0.f; ri = 0.f; } // fix first frame
    x[idx] = xi;
    r[idx] = ri;
}

__global__ void pcgUpdateP(
    float* __restrict__ p,
    const float* __restrict__ z,
    const float* __restrict__ beta_dev,
    int n)
{
    float b = LDG(beta_dev);
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= n) return;

    float pi = z[idx] + b * p[idx];
    if (idx < 6) pi = 0.f;
    p[idx] = pi;
}

__global__ void pcgComputeAlpha(float* alpha, const float* rz, const float* pAp) {
    float denom = LDG(pAp);
    *alpha = (fabsf(denom) > 1e-20f) ? (LDG(rz) / denom) : 0.f;
}
__global__ void pcgComputeBeta(float* beta, const float* rz_new, const float* rz_old) {
    float denom = LDG(rz_old);
    *beta = (fabsf(denom) > 1e-20f) ? (LDG(rz_new) / denom) : 0.f;
}

// ============================================================================
// PCG workspace (reused across GN iterations)
// ============================================================================
struct PCGWorkspace {
    float* r = nullptr;
    float* z = nullptr;
    float* p = nullptr;
    float* Ap = nullptr;

    float* d_rz = nullptr, * d_rz_new = nullptr, * d_pAp = nullptr;
    float* d_alpha = nullptr, * d_beta = nullptr;

    int n = 0;

    void alloc(int n_) {
        n = n_;
        CUDA_CHECK(cudaMalloc(&r, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&z, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&Ap, n * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_rz, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rz_new, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pAp, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta, sizeof(float)));
    }

    void release() {
        cudaFree(r); cudaFree(z); cudaFree(p); cudaFree(Ap);
        cudaFree(d_rz); cudaFree(d_rz_new); cudaFree(d_pAp);
        cudaFree(d_alpha); cudaFree(d_beta);
        r = z = p = Ap = nullptr;
        d_rz = d_rz_new = d_pAp = d_alpha = d_beta = nullptr;
        n = 0;
    }
};

// Optimized PCG: solves H x = b, where H is implicitly defined by (edges, edge_A21).
static inline void pcgSolve_EdgeBlocks(
    cublasHandle_t cublas,
    cudaStream_t stream,
    const cuEdge* d_edges,
    int numEdges,
    const float* d_edgeA21,
    const float* d_Minv36,
    const float* d_b,
    float* d_x,
    int numFrames,
    int maxIters,
    PCGWorkspace& ws)
{
    const int n = 6 * numFrames;
    const int offset = 6;
    const int nVar = n - offset;

    // x = 0
    CUDA_CHECK(cudaMemsetAsync(d_x, 0, n * sizeof(float), stream));

    // r = b
    CUDA_CHECK(cudaMemcpyAsync(ws.r, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // z = M^-1 r
    applyBlockJacobi << <numFrames, 32, 0, stream >> > (d_Minv36, ws.r, ws.z, numFrames);

    // p = z
    CUDA_CHECK(cudaMemcpyAsync(ws.p, ws.z, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // cuBLAS dot products with DEVICE pointer mode
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    CUBLAS_CHECK(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE));

    // rz = r^T z
    CUBLAS_CHECK(cublasSdot(cublas, nVar, ws.r + offset, 1, ws.z + offset, 1, ws.d_rz));

    dim3 block1d(256);
    dim3 grid1d((n + block1d.x - 1) / block1d.x);

    for (int it = 0; it < maxIters; ++it) {
        // Ap = H p
        CUDA_CHECK(cudaMemsetAsync(ws.Ap, 0, n * sizeof(float), stream));
        applyH_fromEdges << <numEdges, 32, 0, stream >> > (d_edges, numEdges, d_edgeA21, ws.p, ws.Ap);

        // pAp = p^T Ap
        CUBLAS_CHECK(cublasSdot(cublas, nVar, ws.p + offset, 1, ws.Ap + offset, 1, ws.d_pAp));

        // alpha = rz / pAp
        pcgComputeAlpha << <1, 1, 0, stream >> > (ws.d_alpha, ws.d_rz, ws.d_pAp);

        // x += alpha p ; r -= alpha Ap
        pcgUpdateXR << <grid1d, block1d, 0, stream >> > (d_x, ws.r, ws.p, ws.Ap, ws.d_alpha, n);

        // z = M^-1 r
        applyBlockJacobi << <numFrames, 32, 0, stream >> > (d_Minv36, ws.r, ws.z, numFrames);

        // rz_new = r^T z
        CUBLAS_CHECK(cublasSdot(cublas, nVar, ws.r + offset, 1, ws.z + offset, 1, ws.d_rz_new));

        // beta = rz_new / rz
        pcgComputeBeta << <1, 1, 0, stream >> > (ws.d_beta, ws.d_rz_new, ws.d_rz);

        // p = z + beta p
        pcgUpdateP << <grid1d, block1d, 0, stream >> > (ws.p, ws.z, ws.d_beta, n);

        // rz = rz_new
        CUDA_CHECK(cudaMemcpyAsync(ws.d_rz, ws.d_rz_new, sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
}

// ============================================================================
// Exported API: full optimized Dense BA
// ============================================================================
extern "C" void denseOptPoseSE3PCG(
    cv::Mat mK, cv::Mat mDistCoeffs, cv::Mat Raymap,
    std::vector<ORB_SLAM3::KeyFrame*> vpKFs,
    std::vector<std::pair<ORB_SLAM3::KeyFrame*, Eigen::Matrix4f>>&mvTwc_KF_opt,
    int maxIter,
    float tol)
{
    using std::cout;
    using std::endl;

    // -------- Data filtering (match original intent) --------
    cout << "[Dense BA] Filtering keyframe data..." << endl;
    std::sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);

    auto it = vpKFs.begin();
    while (it != vpKFs.end())
    {
        ORB_SLAM3::KeyFrame* pKF = *it;
        if (pKF->isBad() || !pKF->hasValidMSTransform || pKF->depth_KF.empty() || pKF->color_KF.empty()) {
            it = vpKFs.erase(it);
            continue;
        }
        std::vector<ORB_SLAM3::KeyFrame*> vCov = pKF->GetBestCovisibilityKeyFrames(1);
        if (vCov.empty() || vCov[0]->isBad() || !vCov[0]->hasValidMSTransform ||
            vCov[0]->depth_KF.empty() || vCov[0]->color_KF.empty())
        {
            it = vpKFs.erase(it);
            continue;
        }
        ++it;
    }

    if (vpKFs.size() < 2) {
        mvTwc_KF_opt.clear();
        for (auto* kf : vpKFs) mvTwc_KF_opt.emplace_back(kf, kf->GetPoseInverse().matrix());
        cout << "[Dense BA] Not enough valid keyframes." << endl;
        return;
    }

    // -------- Camera / raymap --------
    int width = Raymap.cols, height = Raymap.rows;
    size_t imgSize = size_t(width) * size_t(height);

    cuCam h_cam{};
    h_cam.width = width; h_cam.height = height;
    h_cam.fx = mK.at<double>(0, 0); h_cam.fy = mK.at<double>(1, 1);
    h_cam.cx = mK.at<double>(0, 2); h_cam.cy = mK.at<double>(1, 2);
    h_cam.k1 = (double)mDistCoeffs.at<double>(0);
    h_cam.k2 = (double)mDistCoeffs.at<double>(1);
    h_cam.p1 = (double)mDistCoeffs.at<double>(2);
    h_cam.p2 = (double)mDistCoeffs.at<double>(3);
    h_cam.k3 = (double)mDistCoeffs.at<double>(4);

    double2* d_raymap = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raymap, imgSize * sizeof(double2)));
    CUDA_CHECK(cudaMemcpy(d_raymap, Raymap.ptr<double>(), imgSize * sizeof(double2), cudaMemcpyHostToDevice));

    // -------- Upload depth/normal maps for each KF --------
    const int numFrames = (int)vpKFs.size();

    std::vector<float*>  d_I(numFrames, nullptr), d_Igx(numFrames, nullptr), d_Igy(numFrames, nullptr);
    std::vector<float*>  d_depth(numFrames, nullptr);
    std::vector<float3*> d_normal(numFrames, nullptr);
    std::vector<uint8_t*>  d_mask(numFrames, nullptr);

    for (int iF = 0; iF < numFrames; ++iF)
    {
        ORB_SLAM3::KeyFrame* pKF = vpKFs[iF];

        cv::Mat I, Igx, Igy, mask;
        pKF->color_KF.convertTo(I, CV_32F, 1.0 / 255.0);
        cv::Scharr(I, Igx, CV_32F, 1, 0); cv::blur(Igx, Igx, cv::Size(3, 3));
        cv::Scharr(I, Igy, CV_32F, 0, 1); cv::blur(Igy, Igy, cv::Size(3, 3));
        /*cv::Mat I_ft, Igx_ft, Igy_ft;
        I_ft = cv::Mat::zeros(I.size(), I.type()); I.copyTo(I_ft, pKF->mask_KF);
        Igx_ft = cv::Mat::zeros(I.size(), I.type()); Igx.copyTo(Igx_ft, pKF->mask_KF);
        Igy_ft = cv::Mat::zeros(I.size(), I.type()); Igy.copyTo(Igy_ft, pKF->mask_KF);*/

        CUDA_CHECK(cudaMalloc(&d_I[iF], imgSize * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_I[iF], I.ptr<float>(),
            imgSize * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_Igx[iF], imgSize * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Igx[iF], Igx.ptr<float>(),
            imgSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_Igy[iF], imgSize * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Igy[iF], Igy.ptr<float>(),
            imgSize * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_depth[iF], imgSize * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_depth[iF], pKF->depth_KF.ptr<float>(),
            imgSize * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_normal[iF], imgSize * sizeof(float3)));
        // normal_KF is CV_32FC3 => contiguous float[3] per pixel; matches float3 layout
        CUDA_CHECK(cudaMemcpy(d_normal[iF], pKF->normal_KF.ptr<float>(),
            imgSize * sizeof(float3), cudaMemcpyHostToDevice));

        mask = pKF->mask_KF.isContinuous() ? pKF->mask_KF : pKF->mask_KF.clone();
        CUDA_CHECK(cudaMalloc(&d_mask[iF], imgSize * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_mask[iF], mask.ptr<uint8_t>(),
            imgSize * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    // -------- Build KF pointer -> index map (accelerate std::find) --------
    std::unordered_map<ORB_SLAM3::KeyFrame*, int> kfIndex;
    kfIndex.reserve(numFrames * 2);
    for (int i = 0; i < numFrames; ++i) kfIndex[vpKFs[i]] = i;

    // -------- Build edges (same selection as original: best covisibility K=5) --------
    std::vector<cuEdge> h_edges;
    h_edges.reserve(numFrames * 5);

    for (int i = 0; i < numFrames; ++i)
    {
        ORB_SLAM3::KeyFrame* pKF = vpKFs[i];
        std::vector<ORB_SLAM3::KeyFrame*> vCov = pKF->GetBestCovisibilityKeyFrames(5);

        for (auto* cov : vCov)
        {
            auto itj = kfIndex.find(cov);
            if (itj == kfIndex.end()) continue;
            int j = itj->second;
            if (j == i) continue;
            if (cov->isBad() || !cov->hasValidMSTransform || cov->depth_KF.empty()) continue;

            cuEdge e{};
            e.i = i;
            e.j = j;
            // geo stuff
            e.depth_i = d_depth[i];
            e.depth_j = d_depth[j];
            e.normal_i = d_normal[i];
            // pho stuff
            e.I_i = d_I[i];
            e.I_j = d_I[j];
            e.I_i_gx = d_Igx[i];
            e.I_i_gy = d_Igy[i];
            e.mask_j = d_mask[j];

            // Relative pose: T_{i<-j} = Tcw_i * Twc_j (Twc = GetPoseInverse)
            Eigen::Matrix4f Twc_i = pKF->GetPoseInverse().matrix();
            Eigen::Matrix4f Twc_j = cov->GetPoseInverse().matrix();
            Eigen::Matrix4f Tcw_i = Twc_i.inverse();
            Eigen::Matrix4f Tij = Tcw_i * Twc_j;

            Eigen::Matrix3f R = Tij.block<3, 3>(0, 0);
            Eigen::Vector3f t = Tij.block<3, 1>(0, 3);
            Eigen::Matrix3f R_inv = R.transpose();
            Eigen::Vector3f t_inv = -R_inv * t;

            cuRelPose pose{};
            for (int rr = 0; rr < 3; ++rr) {
                for (int cc = 0; cc < 3; ++cc) {
                    pose.R[rr * 3 + cc] = R(rr, cc);
                    pose.R_inv[rr * 3 + cc] = R_inv(rr, cc);
                }
                pose.t[rr] = t(rr);
                pose.t_inv[rr] = t_inv(rr);
            }
            e.T_ij = pose;

            // Ad(Tcw_i): [R, skew(t)R; 0, R]  (same as your original)
            Eigen::Matrix<float, 6, 6> Ad;
            Ad.setZero();
            Eigen::Matrix3f ad_R = Tcw_i.block<3, 3>(0, 0);
            Eigen::Vector3f ad_t = Tcw_i.block<3, 1>(0, 3);
            Ad.block<3, 3>(0, 0) = ad_R;
            Ad.block<3, 3>(0, 3) = skew_symmetric(ad_t) * ad_R;
            Ad.block<3, 3>(3, 3) = ad_R;

            cuAdj AdInv{};
            std::memcpy(AdInv.data, Ad.data(), 36 * sizeof(float));
            e.d_AdInv = AdInv;

            h_edges.push_back(e);
        }
    }

    if (h_edges.empty()) {
        mvTwc_KF_opt.clear();
        for (int i = 0; i < numFrames; ++i) {
            mvTwc_KF_opt.emplace_back(vpKFs[i], vpKFs[i]->GetPoseInverse().matrix());
        }
        std::cerr << "[Dense BA] No edges constructed." << std::endl;

        // Early cleanup (no device edges/system buffers were allocated yet).
        cudaFree(d_raymap);
        for (int i = 0; i < numFrames; ++i) {
            cudaFree(d_depth[i]);
            cudaFree(d_normal[i]);
            cudaFree(d_I[i]);
            cudaFree(d_Igx[i]);
            cudaFree(d_Igy[i]);
            cudaFree(d_mask[i]);
        }
        return;
    }

    // -------- Device edges --------
    const int numEdges = (int)h_edges.size();

    cuEdge* d_edges = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edges, numEdges * sizeof(cuEdge)));
    CUDA_CHECK(cudaMemcpy(d_edges, h_edges.data(), numEdges * sizeof(cuEdge), cudaMemcpyHostToDevice));

    // -------- Allocate system buffers --------
    const int n = 6 * numFrames;

    float* d_edge_g = nullptr; // [E,6]
    float* d_edge_A21 = nullptr; // [E,21]
    float* d_b = nullptr; // [F,6]
    float* d_Hii21 = nullptr; // [F,21]
    float* d_Minv36 = nullptr; // [F,36]
    float* d_delta = nullptr; // [F,6] solution

    // Residual monitoring (global, per GN iteration)
    float* d_geo_sumsq = nullptr;           // sum of e_geo^2 (UNWEIGHTED)
    unsigned int* d_geo_count = nullptr;    // count of valid geo residuals
    float* d_pho_sumsq = nullptr;           // sum of e_pho^2 (UNWEIGHTED)
    unsigned int* d_pho_count = nullptr;    // count of valid pho residuals

    CUDA_CHECK(cudaMalloc(&d_edge_g, numEdges * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edge_A21, numEdges * 21 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Hii21, numFrames * 21 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Minv36, numFrames * 36 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta, n * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_geo_sumsq, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_geo_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_pho_sumsq, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pho_count, sizeof(unsigned int)));

    // PCG workspace
    PCGWorkspace pcgWS;
    pcgWS.alloc(n);

    // cuBLAS + stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    dim3 block2d(kBlockX, kBlockY);
    dim3 grid2d((width + kBlockX - 1) / kBlockX,
        (height + kBlockY - 1) / kBlockY,
        numEdges);

    // Scatter kernels use <=32 threads
    dim3 block32(32);

    dim3 blockInv(256);
    dim3 gridInv((numFrames + blockInv.x - 1) / blockInv.x);

    dim3 block1d(256);
    dim3 grid1d((n + block1d.x - 1) / block1d.x);

    std::vector<Eigen::Matrix4f> vTwc_opt(numFrames);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // Host-side residual monitoring outputs (updated every GN iteration)
    float h_geo_sumsq = 0.f;
    unsigned int h_geo_count = 0u;
    float h_pho_sumsq = 0.f;
    unsigned int h_pho_count = 0u;
    // -------- GN outer loop --------
    for (int iter = 0; iter < maxIter; ++iter)
    {
        // Update per-edge T_ij and AdInv from current KF poses (host), then copy to device.
        for (int eIdx = 0; eIdx < numEdges; ++eIdx) {
            cuEdge& e = h_edges[eIdx];

            ORB_SLAM3::KeyFrame* kfi = vpKFs[e.i];
            ORB_SLAM3::KeyFrame* kfj = vpKFs[e.j];

            Eigen::Matrix4f Twc_i = kfi->GetPoseInverse().matrix();
            Eigen::Matrix4f Twc_j = kfj->GetPoseInverse().matrix();
            Eigen::Matrix4f Tcw_i = Twc_i.inverse();
            Eigen::Matrix4f Tij = Tcw_i * Twc_j;

            Eigen::Matrix3f R = Tij.block<3, 3>(0, 0);
            Eigen::Vector3f t = Tij.block<3, 1>(0, 3);
            Eigen::Matrix3f R_inv = R.transpose();
            Eigen::Vector3f t_inv = -R_inv * t;

            cuRelPose pose{};
            for (int rr = 0; rr < 3; ++rr) {
                for (int cc = 0; cc < 3; ++cc) {
                    pose.R[rr * 3 + cc] = R(rr, cc);
                    pose.R_inv[rr * 3 + cc] = R_inv(rr, cc);
                }
                pose.t[rr] = t(rr);
                pose.t_inv[rr] = t_inv(rr);
            }
            e.T_ij = pose;

            // Ad(Tcw_i)
            Eigen::Matrix<float, 6, 6> Ad;
            Ad.setZero();
            Eigen::Matrix3f ad_R = Tcw_i.block<3, 3>(0, 0);
            Eigen::Vector3f ad_t = Tcw_i.block<3, 1>(0, 3);
            Ad.block<3, 3>(0, 0) = ad_R;
            Ad.block<3, 3>(0, 3) = skew_symmetric(ad_t) * ad_R;
            Ad.block<3, 3>(3, 3) = ad_R;

            cuAdj AdInv{};
            std::memcpy(AdInv.data, Ad.data(), 36 * sizeof(float));
            e.d_AdInv = AdInv;
        }

        float t = (float)iter / (maxIter - 1);
        float kPhoGain_exp_dec = kPhoGain * powf(1.0f / kPhoGain, t);

        CUDA_CHECK(cudaMemcpyAsync(d_edges, h_edges.data(), numEdges * sizeof(cuEdge),
            cudaMemcpyHostToDevice, stream));

        // 1) edge normal equations
        CUDA_CHECK(cudaMemsetAsync(d_edge_g, 0, numEdges * 6 * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_edge_A21, 0, numEdges * 21 * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_geo_sumsq, 0, sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_geo_count, 0, sizeof(unsigned int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_pho_sumsq, 0, sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_pho_count, 0, sizeof(unsigned int), stream));
        buildEdgeNormalEquations << <grid2d, block2d, 0, stream >> > (
            d_edges, numEdges, h_cam, d_raymap, d_edge_g, d_edge_A21,
            kWGeo, kWPho, kPhoGain_exp_dec,
            d_geo_sumsq, d_geo_count,
            d_pho_sumsq, d_pho_count);
        CUDA_CHECK(cudaGetLastError());

        // Read back residual stats (async; will be valid after the later stream sync)
        CUDA_CHECK(cudaMemcpyAsync(&h_geo_sumsq, d_geo_sumsq, sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&h_geo_count, d_geo_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&h_pho_sumsq, d_pho_sumsq, sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&h_pho_count, d_pho_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

        // 2) global b and Hii
        CUDA_CHECK(cudaMemsetAsync(d_b, 0, n * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_Hii21, 0, numFrames * 21 * sizeof(float), stream));
        scatterEdgesToSystem << <numEdges, block32, 0, stream >> > (
            d_edges, numEdges, d_edge_g, d_edge_A21, d_b, d_Hii21);
        CUDA_CHECK(cudaGetLastError());

        // 3) block-Jacobi preconditioner
        invertBlockJacobi6x6 << <gridInv, blockInv, 0, stream >> > (d_Hii21, d_Minv36, numFrames, kDamping);
        CUDA_CHECK(cudaGetLastError());

        // 4) PCG solve: H * delta = b
        pcgSolve_EdgeBlocks(cublas, stream, d_edges, numEdges, d_edge_A21, d_Minv36,
            d_b, d_delta, numFrames, kPCGIters, pcgWS);

        // Bring delta back (small) and update KF poses on CPU (as in original)
        std::vector<float> h_delta(n);
        CUDA_CHECK(cudaMemcpyAsync(h_delta.data(), d_delta, n * sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Convergence check on delta norm (excluding fixed first frame)
        double norm2 = 0.0;
        for (int i = 6; i < n; ++i) norm2 += double(h_delta[i]) * double(h_delta[i]);
        double stepNorm = std::sqrt(norm2);

        // Residual monitoring at the current linearization point (after edge build)
        double geo_rms = 0.0;
        if (h_geo_count > 0u) {
            geo_rms = std::sqrt(double(h_geo_sumsq) / double(h_geo_count));
        }
        double pho_rms = 0.0;
        if (h_pho_count > 0u) {
            pho_rms = std::sqrt(double(h_pho_sumsq) / double(h_pho_count));
        }

        for (int i = 0; i < numFrames; ++i)
        {
            float vx = h_delta[6 * i + 0];
            float vy = h_delta[6 * i + 1];
            float vz = h_delta[6 * i + 2];
            float wx = h_delta[6 * i + 3];
            float wy = h_delta[6 * i + 4];
            float wz = h_delta[6 * i + 5];

            Eigen::Vector3f v(vx, vy, vz);
            Eigen::Vector3f w(wx, wy, wz);

            float theta = w.norm();
            Eigen::Matrix3f R_inc = Eigen::Matrix3f::Identity();
            Eigen::Vector3f t_inc = v;

            if (theta > 1e-12f) {
                Eigen::Vector3f axis = w / theta;
                Eigen::AngleAxisf aa(theta, axis);
                R_inc = aa.toRotationMatrix();
            }

            ORB_SLAM3::KeyFrame* pKF = vpKFs[i];

            Eigen::Matrix4f Twc_i = pKF->GetPoseInverse().matrix();

            Eigen::Matrix4f T_inc = Eigen::Matrix4f::Identity();
            T_inc.block<3, 3>(0, 0) = R_inc;
            T_inc.block<3, 1>(0, 3) = t_inc;

            Eigen::Matrix4f Twc_new = T_inc * Twc_i;
            vTwc_opt[i] = Twc_new;

            // Write back to KeyFrame for next iter
            Eigen::Matrix4f Tcw_new = Twc_new.inverse();
            Sophus::SE3f Tcw_se3(Tcw_new.block<3, 3>(0, 0), Tcw_new.block<3, 1>(0, 3));
            pKF->SetPose(Tcw_se3);
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t_opt = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        cout << "[Dense BA] iter " << iter
            << " stepNorm=" << stepNorm
            << " geo_rms=" << geo_rms / kDepthScale << " mm"
            << " geo_cnt=" << h_geo_count
            << " pho_rms=" << pho_rms /*/ kPhoGain_exp_dec*/
            << " pho_cnt=" << h_pho_count
            << " time=" << t_opt << endl;
        if ((float)stepNorm < tol) {
            cout << "[Dense BA] Converged (stepNorm < tol)." << endl;
            break;
        }
    }

    // Output poses
    mvTwc_KF_opt.clear();
    for (int i = 0; i < numFrames; ++i) {
        mvTwc_KF_opt.emplace_back(vpKFs[i], vTwc_opt[i]);
    }

    // -------- Cleanup --------
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaStreamDestroy(stream));

    pcgWS.release();

    cudaFree(d_edges);
    cudaFree(d_edge_g);
    cudaFree(d_edge_A21);
    cudaFree(d_b);
    cudaFree(d_Hii21);
    cudaFree(d_Minv36);
    cudaFree(d_delta);
    cudaFree(d_geo_sumsq);
    cudaFree(d_geo_count);
    cudaFree(d_pho_sumsq);
    cudaFree(d_pho_count);
cleanup:
    cudaFree(d_raymap);
    for (int i = 0; i < numFrames; ++i) {
        cudaFree(d_depth[i]);
        cudaFree(d_normal[i]);
        cudaFree(d_I[i]);
        cudaFree(d_Igx[i]);
        cudaFree(d_Igy[i]);
        cudaFree(d_mask[i]);
    }
}
