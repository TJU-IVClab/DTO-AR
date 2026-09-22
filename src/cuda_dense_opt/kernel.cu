
#include "utility.h"
#include "dense_opt.h"
//#include "sparse_opt.h"

/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
*/

int main(int argc, char** argv)
{
	const string pathToPoses = R"(D:\ins_att\ORB_SLAM3-master\bin\KeyFrameTrajectory.txt)";
	//const string pathToPoses = R"(C:\Users\ST23re\Desktop\sparse_refined.txt)";
	const string pathToSrc = R"(D:\ins_att\MatteSpot\data\dataset)";

	const unsigned int w = 1120, h = 800;
	double fx = 1154.93, fy = 1154.95, cx = 560.74, cy = 400.553;
	double k1 = -0.0867707, k2 = 0.147489, k3 = -0.0370996;
	double p1 = 0.000417617, p2 = 0.000554355;
	const cv::Mat mK = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
	const cv::Mat mDistCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);

	cv::Mat map1, map2;
	cv::initUndistortRectifyMap(
		mK, mDistCoeffs,			// 相机内参与畸变系数
		cv::Mat::eye(3, 3, CV_64F), // 复位变换矩阵 (identity 或 R)
		mK,							// 校正后的投影矩阵 (可与 K 相同)
		cv::Size(w, h),             // 图像尺寸
		CV_32FC1,					// 输出 map 类型
		map1, map2);

	// -------- prefix end --------
	std::vector<Eigen::Matrix4f> poses, poses_opt;
	if (!loadCameraPoses(pathToPoses, poses)) return -1;
	poses_opt = poses;
	vector<cv::Mat> imGs, imDs, imNs;
	if (!loadRGBDNormal(pathToSrc, map1, map2, imGs, imDs, imNs)) return -1;

	size_t nframes = poses.size();
	if (nframes != imGs.size()) { cout << "Inconsistent Input!" << endl; return -1; }
	else cout << nframes << "frames Loaded!" << endl;
	vector<cv::Point3f> points;
	vector<cv::Vec3b> colors;
	for (int i = 0; i < nframes - 1; i++)
	{
		// initial pose => T_ij = inv(T_i) * T_j
		Eigen::Matrix4f T_i = poses[i], T_j = poses[i + 1];
		Eigen::Matrix4f T_ij = T_i.inverse() * T_j;
		//T_ij.block<3, 1>(0, 3) *= 1.01; // test pho converge

		// optimize
		cv::Mat im1, im2;
		im1 = ExtractMatteSpots(imGs[i]);
		cv::Mat grad_x_scharr, grad_y_scharr;
		cv::Scharr(im1, grad_x_scharr, CV_32F, 1, 0); cv::blur(grad_x_scharr, grad_x_scharr, cv::Size(3, 3));
		cv::Scharr(im1, grad_y_scharr, CV_32F, 0, 1); cv::blur(grad_y_scharr, grad_y_scharr, cv::Size(3, 3));
		im2 = ExtractMatteSpots(imGs[i + 1]);

		Eigen::Matrix4f T_opt = denseOptPoseSE3(
			im1, grad_x_scharr, grad_y_scharr, imDs[i], imNs[i],
			im2, imDs[i + 1],
			T_ij
		);

		// update
		poses_opt[i + 1] = poses_opt[i] * T_opt;

		// render
		if (i == 0) textureDepthTo3D(imGs[i], imDs[i], poses_opt[0], points, colors);
		textureDepthTo3D(imGs[i + 1], imDs[i + 1], poses_opt[i + 1], points, colors);
	}

	// 输出优化后的位姿序列
	std::ofstream fout("optimized_poses.txt");
	for (const auto& T : poses_opt) {
		for (int r = 0; r < 4; ++r) {
			for (int c = 0; c < 4; ++c) {
				fout << T(r, c) << " ";
			}
			fout << "\n";
		}
	}
	fout.close();

	saveModeltoPly(points, colors);

	return 0;
}

//int main(int argc, char** argv)
//{
//	const string pathToSrc = R"(D:\ins_att\ORB_SLAM3-master\evaluation\dataset_2025_11_16)";
//	//const string pathToSrc = R"(D:\ins_att\MatteSpot\data\dataset)";
//
//	const unsigned int w = 1680, h = 1200;
//	double fx = 1732.39, fy = 1732.43, cx = 841.36, cy = 601.08;
//	//const unsigned int w = 1120, h = 800;
//	//double fx = 1154.93, fy = 1154.95, cx = 560.74, cy = 400.553;
//	double k1 = -0.0867707, k2 = 0.147489, k3 = -0.0370996;
//	double p1 = 0.000417617, p2 = 0.000554355;
//	const cv::Mat mK = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
//	const cv::Mat mDistCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
//
//	cv::Mat map1, map2;
//	cv::initUndistortRectifyMap(
//		mK, mDistCoeffs,			// 相机内参与畸变系数
//		cv::Mat::eye(3, 3, CV_64F), // 复位变换矩阵 (identity 或 R)
//		mK,							// 校正后的投影矩阵 (可与 K 相同)
//		cv::Size(w, h),             // 图像尺寸
//		CV_32FC1,					// 输出 map 类型
//		map1, map2);
//
//	// -------- prefix end --------
//	vector<cv::Mat> imGs, imDs, imNs;
//	if (!loadRGBDNormal(pathToSrc, map1, map2, imGs, imDs, imNs)) return -1;
//
//	size_t nframes = imGs.size();
//	if (nframes != imDs.size()) { cout << "Inconsistent Input!" << endl; return -1; }
//	else cout << nframes << "frames Loaded!" << endl;
//	vector<cv::Point3f> points;
//	vector<cv::Vec3b> colors;
//	cv::namedWindow("canvas", cv::WINDOW_NORMAL);
//	cv::resizeWindow("canvas", cv::Size(1680 / 2, 1200 / 2));
//	for (int i = 0; i < nframes - 1; i++)
//	{
//		// optimize
//		cv::Mat im1, im2;
//		im1 = ExtractMatteSpots(imGs[i]);
//		cv::imshow("canvas", im1);
//
//		cv::waitKey(0);
//		//cv::Mat grad_x_scharr, grad_y_scharr;
//		//cv::Scharr(im1, grad_x_scharr, CV_32F, 1, 0); cv::blur(grad_x_scharr, grad_x_scharr, cv::Size(3, 3));
//		//cv::Scharr(im1, grad_y_scharr, CV_32F, 0, 1); cv::blur(grad_y_scharr, grad_y_scharr, cv::Size(3, 3));
//		//im2 = ExtractMatteSpots(imGs[i + 1]);
//
//	}
//
//	return 0;
//}