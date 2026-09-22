#include "Scanner.h"

namespace ORB_SLAM3 {
	PhotoneoSensor::PhotoneoSensor(Settings* settings)
	{
		// initialize distortion stuff
		mK = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
		mDistCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
		// Approach A: (or simply use the calibrated map from photoneo device)
		// Tips: Used to restore depth maps to point clouds, avoiding interpolation on the depth map.
		vector<cv::Point2d> uv, RayVec;
		for (int v = 0; v < imgHeight; ++v)
			for (int u = 0; u < imgWidth; ++u)
				uv.push_back(cv::Point2d(u, v));
		cv::undistortPoints(uv, RayVec, mK, mDistCoeffs);
		cv::Mat mapAB(imgHeight, imgWidth, CV_64FC2);
		for (int u = 0; u < imgWidth; ++u)
			for (int v = 0; v < imgHeight; ++v)
				mapAB.at<cv::Point2d>(v, u) = RayVec[v * imgWidth + u];
		Raymap = mapAB;
		// Approach B:
		// Tips: Used for anyType image undistortion.
		cv::initUndistortRectifyMap(
			mK, mDistCoeffs,			// 相机内参与畸变系数
			cv::Mat::eye(3, 3, CV_64F), // 复位变换矩阵 (identity 或 R)
			mK,							// 校正后的投影矩阵 (可与 K 相同)
			cv::Size(imgWidth, imgHeight),             // 图像尺寸
			CV_32FC1,					// 输出 map 类型
			map1, map2);

		// photoneo settings
		FPS = settings->fps();
		LEDPower = settings->LEDPower();
		Confidence = settings->Confidence();
	}

	PhotoneoSensor::~PhotoneoSensor() {
		CorrectDisconnectExample();
		if (mFrame)
		{
			delete mFrame;
			mFrame = nullptr;
		}
		if (mpcdHandler)
		{
			delete mpcdHandler;
			mpcdHandler = nullptr;
		}
	}
	void PhotoneoSensor::GetAvailableDevicesExample() {
		// Wait for the PhoXiControl
		while (!Factory.isPhoXiControlRunning()) {
			LOCAL_CROSS_SLEEP(100);
		}
		cout << "PhoXi Control Version: " << Factory.GetPhoXiControlVersion()
			<< endl;
		cout << "PhoXi API Version: " << Factory.GetAPIVersion() << endl;

		DeviceList = Factory.GetDeviceList();
		cout << "PhoXi Factory found " << DeviceList.size() << " devices."
			<< endl
			<< endl;
		pho::api::PhoXiDeviceInformation* DeviceInfo;
		for (size_t i = 0; i < DeviceList.size(); ++i) {
			DeviceInfo = &DeviceList[i];
			cout << "Device: " << i << endl;
			cout << "  Name:                    " << DeviceInfo->Name << endl;
			cout << "  Hardware Identification: " << DeviceInfo->HWIdentification << endl;
			cout << "  Type:                    " << string(DeviceInfo->Type) << endl;
			cout << "  Firmware version:        " << DeviceInfo->FirmwareVersion << endl;
			cout << "  Variant:                 " << DeviceInfo->Variant << endl;
			cout << "  IsFileCamera:            " << (DeviceInfo->IsFileCamera ? "Yes" : "No") << endl;
			cout << "  Feature-Alpha:           " << (DeviceInfo->CheckFeature("Alpha") ? "Yes" : "No") << endl;
			cout << "  Feature-Color:           " << (DeviceInfo->CheckFeature("Color") ? "Yes" : "No") << endl;
			cout << "  Status:                  "
				<< (DeviceInfo->Status.Attached
					? "Attached to PhoXi Control. "
					: "Not Attached to PhoXi Control. ")
				<< (DeviceInfo->Status.Ready ? "Ready to connect"
					: "Occupied")
				<< endl
				<< endl;
		}
	}
	void PhotoneoSensor::ConnectPhoXiDeviceBySerialExample() {
		/*cout << endl
			<< "Please enter the Hardware Identification Number: ";*/
		string HardwareIdentification = "CTR-095";
		/*if (!ReadLine(HardwareIdentification)) {
			cout << "Incorrect input!" << endl;
			return;
		}*/
		pho::api::PhoXiTimeout Timeout = pho::api::PhoXiTimeout::ZeroTimeout;
		PhoXiDevice = Factory.CreateAndConnect(HardwareIdentification, Timeout);
		if (PhoXiDevice) {
			cout << "Connection to the device " << HardwareIdentification
				<< " was Successful!" << endl;
		}
		else {
			cout << "Connection to the device " << HardwareIdentification
				<< " was Unsuccessful!" << endl;
		}
	}

	void PhotoneoSensor::PrintFrameInfo(const pho::api::PFrame& Frame) {
		const pho::api::FrameInfo& FrameInfo = Frame->Info;
		cout << "  Frame params: " << endl;
		cout << "    Frame Index: " << FrameInfo.FrameIndex << endl;
		cout << "    Frame Timestamp: " << FrameInfo.FrameTimestamp << " ms"
			<< endl;
		cout << "    Frame Acquisition duration: " << FrameInfo.FrameDuration
			<< " ms" << endl;
		cout << "    Frame Computation duration: "
			<< FrameInfo.FrameComputationDuration << " ms" << endl;
		cout << "    Frame Transfer duration: "
			<< FrameInfo.FrameTransferDuration << " ms" << endl;
		cout << "    Sensor Position: [" << FrameInfo.SensorPosition.x << "; "
			<< FrameInfo.SensorPosition.y << "; "
			<< FrameInfo.SensorPosition.z << "]" << endl;
		cout << "    Total scan count: " << FrameInfo.TotalScanCount << endl;
		cout << "    Color Camera Position: [" << FrameInfo.ColorCameraPosition.x << "; "
			<< FrameInfo.ColorCameraPosition.y << "; "
			<< FrameInfo.ColorCameraPosition.z << "]" << endl;
	}
	void PhotoneoSensor::PrintFrameData(const pho::api::PFrame& Frame) {
		if (Frame->Empty()) {
			cout << "Frame is empty.";
			return;
		}
		cout << "  Frame data: " << endl;
		if (!Frame->PointCloud.Empty()) {
			cout << "    PointCloud:    (" << Frame->PointCloud.Size.Width
				<< " x " << Frame->PointCloud.Size.Height
				<< ") Type: " << Frame->PointCloud.GetElementName()
				<< endl;
		}
		if (!Frame->NormalMap.Empty()) {
			cout << "    NormalMap:     (" << Frame->NormalMap.Size.Width
				<< " x " << Frame->NormalMap.Size.Height
				<< ") Type: " << Frame->NormalMap.GetElementName()
				<< endl;
		}
		if (!Frame->DepthMap.Empty()) {
			cout << "    DepthMap:      (" << Frame->DepthMap.Size.Width
				<< " x " << Frame->DepthMap.Size.Height
				<< ") Type: " << Frame->DepthMap.GetElementName()
				<< endl;
		}
		if (!Frame->ConfidenceMap.Empty()) {
			cout << "    ConfidenceMap: (" << Frame->ConfidenceMap.Size.Width
				<< " x " << Frame->ConfidenceMap.Size.Height
				<< ") Type: " << Frame->ConfidenceMap.GetElementName()
				<< endl;
		}
		if (!Frame->Texture.Empty()) {
			cout << "    Texture:       (" << Frame->Texture.Size.Width
				<< " x " << Frame->Texture.Size.Height
				<< ") Type: " << Frame->Texture.GetElementName() << endl;
		}
		if (!Frame->TextureRGB.Empty()) {
			cout << " TextureRGB:       (" << Frame->TextureRGB.Size.Width
				<< " x " << Frame->TextureRGB.Size.Height
				<< ") Type: " << Frame->TextureRGB.GetElementName() << endl;
		}
		if (!Frame->ColorCameraImage.Empty()) {
			cout << " ColorCameraImage:       (" << Frame->ColorCameraImage.Size.Width
				<< " x " << Frame->ColorCameraImage.Size.Height
				<< ") Type: " << Frame->ColorCameraImage.GetElementName() << endl;
		}
	}
	void PhotoneoSensor::DataHandlingExample() {
		// Check if we have SampleFrame Data
		if (!SampleFrame || SampleFrame->Empty()) {
			cout << "Frame does not exist, or has no content!" << endl;
			return;
		}

		// We will count the number of measured points
		if (!SampleFrame->PointCloud.Empty()) {
			int MeasuredPoints = 0;
			pho::api::Point3_32f ZeroPoint(0.0f, 0.0f, 0.0f);
			for (int y = 0; y < SampleFrame->PointCloud.Size.Height; ++y) {
				for (int x = 0; x < SampleFrame->PointCloud.Size.Width; ++x) {
					if (SampleFrame->PointCloud[y][x] != ZeroPoint) {
						MeasuredPoints++;
					}
				}
			}
			cout << "Your sample PointCloud has " << MeasuredPoints
				<< " measured points." << endl;

			float* MyLocalCopy =
				new float[SampleFrame->PointCloud.GetElementsCount() * 3];

			pho::api::Point3_32f* RawPointer = SampleFrame->PointCloud.GetDataPtr();
			memcpy(MyLocalCopy, RawPointer, SampleFrame->PointCloud.GetDataSize());
			// Data are organized as a matrix of X, Y, Z floats, see the
			// documentation for all other types

			delete[] MyLocalCopy;
			// Data from SampleFrame, or all other frames that are returned by the
			// device are copied from the Cyclic buffer and will remain in the
			// memory until the Frame will go out of scope You can specifically call
			// SampleFrame->PointCloud.Clear() to release some of the data
		}

		// You can store the Frame as a ply structure
		// If you don't specify Output folder the PLY file will be saved where
		// FullAPIExample_CSharp.exe is
		const auto outputFolder =
			OutputFolder.empty() ? string() : OutputFolder + DELIMITER;
		const auto sampleFramePly = outputFolder + "SampleFrame.ply";
		cout << "Saving frame as 'SampleFrame.ply'" << endl;
		if (SampleFrame->SaveAsPly(sampleFramePly, true, true)) {
			cout << "Saved sample frame as PLY to: " << sampleFramePly
				<< endl;
		}
		else {
			cout << "Could not save sample frame as PLY to " << sampleFramePly
				<< " !" << endl;
		}
		// You can save scans to any format, you only need to specify path + file
		// name API will look at extension and save the scan in the correct format
		// You can define which options to save (PointCloud, DepthMap, ...) in PhoXi
		// Control application -> Saving options, or set options directly from code
		// via optionsl 3rd parameter. This method has a an optional 2nd
		// parameter: FrameId Use this option to save other scans than the last one
		// Absolute path is prefered
		// If you don't specify Output folder the file will be saved to
		// %APPDATA%\PhotoneoPhoXiControl\ folder on Windows or
		// ~/.PhotoneoPhoXiControl/ on Linux
		const auto sampleFrameTiffFormat = outputFolder + "OtherSampleFrame.tif";
		if (PhoXiDevice->SaveLastOutput(sampleFrameTiffFormat)) {
			cout << "Saved sample frame to: " << sampleFrameTiffFormat
				<< endl;
		}
		else {
			cout << "Could not save sample frame to: " << sampleFrameTiffFormat
				<< " !" << endl;
		}
		// Overide saving options
		const auto sampleFramePrawFormat = outputFolder + "OtherSampleFrame.praw";
		const string jsonOptions = R"json({
        "UseCompression": true
    })json";
		if (PhoXiDevice->SaveLastOutput(sampleFramePrawFormat, -1, jsonOptions)) {
			cout << "Saved sample frame to: " << sampleFramePrawFormat
				<< endl;
		}
		else {
			cout << "Could not save sample frame to: " << sampleFramePrawFormat
				<< " !" << endl;
		}

		// If you want OpenCV support, you need to link appropriate libraries and
		// add OpenCV include directory To add the support, add #define
		// PHOXI_OPENCV_SUPPORT before include of PhoXi include files For details
		// check also MinimalOpenCVExample
#ifdef PHOXI_OPENCV_SUPPORT
		if (!SampleFrame->PointCloud.Empty()) {
			cv::Mat PointCloudMat;
			if (SampleFrame->PointCloud.ConvertTo(PointCloudMat)) {
				cv::Point3f MiddlePoint = PointCloudMat.at<cv::Point3f>(
					PointCloudMat.rows / 2, PointCloudMat.cols / 2);
				cout << "Middle point: " << MiddlePoint.x << "; "
					<< MiddlePoint.y << "; " << MiddlePoint.z;
			}
		}
#endif
		// If you want PCL support, you need to link appropriate libraries and add
		// PCL include directory To add the support, add #define PHOXI_PCL_SUPPORT
		// before include of PhoXi include files For details check also
		// MinimalPclExample
#ifdef PHOXI_PCL_SUPPORT
	// The PCL convert will convert the appropriate data into the pcl PointCloud
	// based on the Point Cloud type
		pcl::PointCloud<pcl::PointXYZRGBNormal> MyPCLCloud;
		SampleFrame->ConvertTo(MyPCLCloud);
#endif
	}
	void PhotoneoSensor::FreerunExample()
	{
		//Check if the device is connected
		if (!PhoXiDevice || !PhoXiDevice->isConnected())
		{
			cout << "Device is not created, or not connected!" << endl;
			return;
		}
		//If it is not in Freerun mode, we need to switch the modes
		if (PhoXiDevice->TriggerMode != pho::api::PhoXiTriggerMode::Freerun)
		{
			cout << "Device is not in Freerun mode" << endl;
			if (PhoXiDevice->isAcquiring())
			{
				cout << "Stopping acquisition" << endl;
				//If the device is in Acquisition mode, we need to stop the acquisition
				if (!PhoXiDevice->StopAcquisition())
				{
					throw runtime_error("Error in StopAcquistion");
				}
			}
			cout << "Switching to Freerun mode " << endl;
			//Switching the mode is as easy as assigning of a value, it will call the appropriate calls in the background
			PhoXiDevice->TriggerMode = pho::api::PhoXiTriggerMode::Freerun;
			//Just check if did everything run smoothly
			if (!PhoXiDevice->TriggerMode.isLastOperationSuccessful())
			{
				throw runtime_error(PhoXiDevice->TriggerMode.GetLastErrorMessage().c_str());
			}
		}

		//Start the device acquisition, if necessary
		if (!PhoXiDevice->isAcquiring())
		{
			if (!PhoXiDevice->StartAcquisition())
			{
				throw runtime_error("Error in StartAcquisition");
			}
		}

		//We can clear the current Acquisition buffer -- This will not clear Frames that arrives to the PC after the Clear command is performed
		int ClearedFrames = PhoXiDevice->ClearBuffer();
		cout << ClearedFrames << " were cleared from the cyclic buffer" << endl;

		//While we checked the state of the StartAcquisition call, this check is not necessary, but it is a good practice
		if (!PhoXiDevice->isAcquiring())
		{
			cout << "Device is not acquiring" << endl;
			return;
		}
	}

	void PhotoneoSensor::createFirstConnected() {
		//import PhoXiControl
		try {
			//Connecting to scanner
			ConnectPhoXiDeviceBySerialExample();
			cout << endl;

			//Checks
			//Check if the device is connected
			if (!PhoXiDevice || !PhoXiDevice->isConnected()) {
				cout << "Device is not created, or not connected!" << endl;
				return;
			}

			//Check if the GeneralSettings are Enabled and Can be Set
			if (!PhoXiDevice->MotionCam.isEnabled() || !PhoXiDevice->MotionCam.CanSet() || !PhoXiDevice->MotionCam.CanGet()) {
				cout << "General Settings are not supported by the Device Hardware, or are Read only on the specific device" << endl;
				return;
			}
			//Check if the CapturingModes are Enabled and Can be Set (needed to change the resolution)
			if (!PhoXiDevice->MotionCamCameraMode.isEnabled() || !PhoXiDevice->MotionCamCameraMode.CanGet() || !PhoXiDevice->MotionCamCameraMode.CanSet()) {
				cout << "CamCameraMode Settings are not supported by the Device Hardware, or are Read only on the specific device" << endl;
				return;
			}
			//End Checks

			ChangeMotionCamExample();

			FreerunExample();
		}
		catch (runtime_error& InternalException) {
			cout << endl
				<< "Exception was thrown: " << InternalException.what()
				<< endl;
			if (PhoXiDevice->isConnected()) {
				PhoXiDevice->Disconnect(true);
			}
		}
	}
	void PhotoneoSensor::CorrectDisconnectExample() {
		if (PhoXiDevice) {
			// The whole API is designed on C++ standards, using smart pointers and
			// constructor/destructor logic All resources will be closed automatically,
			// but the device state will not be affected -> it will remain connected in
			// PhoXi Control and if in freerun, it will remain Scanning. To Stop the
			// acquisition, just call
			PhoXiDevice->StopAcquisition();
			PhoXiDevice->Disconnect(false, true);
			// The call PhoXiDevice without Logout will be called automatically by
			// destructor
		}
	}
	void PhotoneoSensor::ChangeMotionCamExample() {
		auto& P = PhoXiDevice->MotionCam;
		P->LaserPower = 4095;
		P->LEDPower = LEDPower;
		P->MaximumFPS = 10; // # 0.25 for debugging
		P->OperationMode = pho::api::PhoXiOperationMode::Camera;

		auto& PCM = PhoXiDevice->MotionCamCameraMode;
		vector<double> SinglePatternExposures = PhoXiDevice->SupportedSinglePatternExposures;
		PCM->Exposure = SinglePatternExposures[0]; // 0 for 10.24
		PCM->OutputTopology = pho::api::PhoXiOutputTopology::FullGrid; // #
		PCM->CodingStrategy = pho::api::PhoXiCodingStrategy::Interreflections;
		PCM->TextureSource = pho::api::PhoXiTextureSource::Laser;

		auto& PCMPS = PhoXiDevice->ProcessingSettings;
		PCMPS->SurfaceSmoothness = pho::api::PhoXiSurfaceSmoothness::Smooth;
		PCMPS->NormalsEstimationRadius = 4;
		PCMPS->PatternCodeCorrection = pho::api::PhoXiPatternCodeCorrection::Strong;
		//PCMPS->PatternDecompositionReach = pho::api::PhoXiPatternDecompositionReach::Large;
		PCMPS->GlareCompensation = true;
		//PCMPS->HoleFilling = pho::api::PhoXiHoleFilling::Medium;
		//PCMPS->CalibrationVolumeOnly = false;
		PCMPS->Confidence = Confidence;// mm
		//Check
		if (!PCMPS.isLastOperationSuccessful())
			throw runtime_error(PCMPS.GetLastErrorMessage().c_str());
	};

	bool PhotoneoSensor::getFrame() {
		try {
			static int index = 0, index_pre = -1;
			SampleFrame = PhoXiDevice->GetFrame(/*You can specify Timeout here -
			default is the Timeout stored in Timeout Feature -> Infinity by default*/);
			index = SampleFrame->Info.FrameIndex;
			//cout << "SampleFrame Index: " << index << endl;
			if (index == index_pre) return false;
			else index_pre = index;
			pho::api::PhoXiSize imSize = SampleFrame->Texture.Size;

			pho::api::Depth_32f* t = SampleFrame->Texture.GetDataPtr(); // float32 C1
			cv::Mat t_f32(imSize.Height, imSize.Width, CV_32FC1, t);
			//t_u8 = new cv::Mat(imSize.Height, imSize.Width, CV_8UC1);
			t_f32.convertTo(mFrame->t_u8, CV_8UC1, 256.0 / 1024); // -> uchar C1
			cv::Mat t_u8c3; // just for pcd rendering
			cv::cvtColor(mFrame->t_u8, t_u8c3, cv::COLOR_GRAY2RGB);
			//cv::imwrite("11.jpg", t_u8);

			pho::api::Depth_32f* d = SampleFrame->DepthMap.GetDataPtr(); // float32 C1
			mFrame->d_f32 = cv::Mat(imSize.Height, imSize.Width, CV_32FC1, d);

			pho::api::Point3_32f* n = SampleFrame->NormalMap.GetDataPtr(); // float32 C3
			mFrame->n_f32 = cv::Mat(imSize.Height, imSize.Width, CV_32FC3, n);

			pho::api::Depth_32f* c = SampleFrame->ConfidenceMap.GetDataPtr(); // float32 C3
			mFrame->c_f32 = cv::Mat(imSize.Height, imSize.Width, CV_32FC1, c);

			/*pho::api::PointCloud32f& pcd = SampleFrame->PointCloud;
			mFrame->pcdPoints.clear();
			mFrame->pcdColors.clear();
			int dSample = 1;
			int ui, vi; // index, which is u(or v) - 1.
			for (int du = 0; du < imSize.Width / dSample; ++du)
				for (int dv = 0; dv < imSize.Height / dSample; ++dv) {
					ui = du * dSample;
					vi = dv * dSample;

					cv::Point2d& xnyn_un = Raymap[vi * imSize.Width + ui];
					float& Z = pcd[vi][ui].z;
					//auto& p = pcd[vi][ui];
					//cv::Point3f point = cv::Point3d(p.x, p.y, p.z) / 1000.0;
					cv::Point3f point = cv::Point3d(xnyn_un.x, xnyn_un.y, 1.0) * Z / 1e3;
					mFrame->pcdPoints.push_back(point); // unit: m

					cv::Vec3b& color = t_u8c3.at<cv::Vec3b>(vi, ui);
					mFrame->pcdColors.push_back(color);
				}*/

			mFrame->frameIndex = index;
			mFrame->frameTimestamp = SampleFrame->Info.FrameTimestamp / 1e3; // unit: s
			return true;
		}
		catch (runtime_error& InternalException) {
			cout << endl
				<< "Exception was thrown: " << InternalException.what()
				<< endl;
			if (PhoXiDevice->isConnected()) {
				PhoXiDevice->Disconnect(true);
			}
			return false;
		}
	}
	void PhotoneoSensor::getFrame_offline(cv::Mat& im, cv::Mat& depthmap, cv::Mat& normalmap, cv::Mat& confidencemap, double& timestamp, int& frameIndex) {
		//cv::Size imSize = im.size();

		if (im.type() == CV_32FC1) im.convertTo(mFrame->t_u8, CV_8UC1, 0.45);
		else if (im.type() == CV_8UC1) mFrame->t_u8 = im;
		else throw runtime_error("Invalid Data Type of Intensity Map!");
		mFrame->d_f32 = depthmap;	// uint: mm
		mFrame->n_f32 = normalmap;
		mFrame->c_f32 = confidencemap;

		/*cv::Mat t_u8c3; // just for pcd rendering
		cv::cvtColor(mFrame->t_u8, t_u8c3, cv::COLOR_GRAY2RGB);

		mFrame->pcdPoints.clear();
		mFrame->pcdColors.clear();
		int dSample = 3;
		int ui, vi; // index, which is u(or v) - 1.
		for (int du = 0; du < imSize.width / dSample; ++du)
			for (int dv = 0; dv < imSize.height / dSample; ++dv) {
				ui = du * dSample;
				vi = dv * dSample;

				cv::Point2d& xnyn_un = Raymap[vi * imSize.width + ui];
				float& Z = depthmap.at<float>(vi, ui);
				//float x_un = (ui - cx) / fx;
				//float y_un = (vi - cy) / fy;
				//cv::Point3f point = cv::Point3d(x_un, y_un, 1.0) * Z / 1000.0;
				cv::Point3f point = cv::Point3d(xnyn_un.x, xnyn_un.y, 1.0) * Z / 1e3;
				mFrame->pcdPoints.push_back(point); // unit: m

				cv::Vec3b& color = t_u8c3.at<cv::Vec3b>(vi, ui);
				mFrame->pcdColors.push_back(color);
			}*/

		mFrame->frameIndex = frameIndex;
		mFrame->frameTimestamp = timestamp;
	}
	void PhotoneoSensor::viewerRun()
	{
		viewer->setUpViewInWindow(1030, 0, 1520, 760);
		viewer->setKeyEventSetsDone(0);
		viewer->addEventHandler(new osgViewer::StatsHandler);

		viewer->setSceneData(mGroup);
		mGroup->setUpdateCallback(meshUpdateCb);
		mGroup->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

		osg::ref_ptr<osgGA::TrackballManipulator> manipulator = new osgGA::TrackballManipulator();
		viewer->setCameraManipulator(manipulator);
		osg::Vec3d eye(0.0, 0.0, -5.0);    // 相机位置
		osg::Vec3d center(0.0, 0.0, 0.0);   // 目标点
		osg::Vec3d up(0.0, -1.0, 0.0);       // 向上方向
		manipulator->setHomePosition(eye, center, up, false);
		viewer->home();

		viewer->run();
	}
	void PhotoneoSensor::renderingPC()
	{
		thread viewerThread(&PhotoneoSensor::viewerRun, this);

		osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();
		geometry->setVertexArray(points);
		geometry->setColorArray(colors);
		geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
		geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 0));
		geometry->setUseVertexBufferObjects(true);

		//// Initialize reconstructor
		//IncrementalTSDFReconstructor reconstructor(0.002, 0.005);
		//open3d::camera::PinholeCameraIntrinsic intrinsic(imgWidth, imgHeight, fx, fy, cx, cy);
		//int idx = 3;

		//chrono::steady_clock::time_point t0, t1, t2;
		//double t_tsdf, t_render;

		while (!checkFinish())
		{
			static size_t localIndex = -1;

			/*size_t frameIndex_KF;
			Eigen::Matrix4f Twc;
			shared_ptr<const cv::Mat> depth, color; // float32, uint8_t
			mpcdHandler->getData(frameIndex_KF, Twc, depth, color);*/
			unique_lock<mutex> lock(mpcdHandler->mutex_pcd);

			if (localIndex != mpcdHandler->frameIndex_KF) localIndex = mpcdHandler->frameIndex_KF;
			else continue;

			auto& depth = mpcdHandler->depth_KF;
			auto& color = mpcdHandler->color_KF;
			auto& Twc = mpcdHandler->Twc_KF;

			if (!depth || !color || depth->empty() || color->empty())
			{
				cerr << "OSG: pcd has no vertex / color !" << endl;
			}
			else
			{
				try
				{
					cv::Size imSize = depth->size();
					int dSample = 3;
					int ui, vi; // index, which is u(or v) - 1.
					for (int du = 0; du < imSize.width / dSample; ++du)
						for (int dv = 0; dv < imSize.height / dSample; ++dv) {
							ui = du * dSample;
							vi = dv * dSample;

							cv::Point2d& xnyn_un = Raymap.at<cv::Point2d>(vi, ui);
							const float Z = depth->at<float>(vi, ui);
							cv::Point3f _p = cv::Point3d(xnyn_un.x, xnyn_un.y, 1.0) * Z / 1e3; // unit: m
							Eigen::Vector4f p = Eigen::Vector4f(_p.x, _p.y, _p.z, 1);
							Eigen::Vector4f pt = Twc * p;
							points->push_back(osg::Vec3(pt.x(), pt.y(), pt.z()));

							const uchar _c = color->at<uchar>(vi, ui);
							colors->push_back(osg::Vec3ub(_c, _c, _c));
						}

					osg::DrawArrays* drawArrays = dynamic_cast<osg::DrawArrays*>(geometry->getPrimitiveSet(0));
					if (drawArrays) {
						drawArrays->setCount(points->size());
					}
					points->dirty(); // force buffer objects to update
					colors->dirty();

					meshUpdateCb->setMesh(geometry);
				}
				catch (runtime_error& InternalException) {
					cout << endl
						<< "Exception was thrown: " << InternalException.what()
						<< endl;
				}
			}

		}

		// Close the viewer after releasing the per-frame data lock.
		if (viewer)
		{
			viewer->setDone(true);
			auto* gw = dynamic_cast<osgViewer::GraphicsWindow*>(
				viewer->getCamera() ? viewer->getCamera()->getGraphicsContext() : nullptr);
			if (gw)
			{
				gw->getEventQueue()->closeWindow();
				gw->getEventQueue()->quitApplication();
			}
		}

		viewerThread.join();
	}
	void PhotoneoSensor::saveModeltoPly(string path)
	{
		cout << "OSG: total vertices → " << points->size() << endl;
		cout << "Saving geometry, this might take a while..." << endl;

		auto now = chrono::system_clock::now();
		time_t now_time_t = chrono::system_clock::to_time_t(now);
		tm now_tm = *localtime(&now_time_t);
		char buffer[100];
		strftime(buffer, sizeof(buffer), "%m_%d_%H-%M-%S", &now_tm);
		string time_str(buffer);

		string filename = path == "" ? ("model_" + time_str + ".ply") : path;
		ofstream outFile(filename, ios::binary);
		if (!outFile.is_open()) {
			cerr << "Error opening file for writing: " << filename << endl;
			return;
		}

		if (!points || points->empty()) {
			cerr << "No points to save." << endl;
			return;
		}
		if (!colors || colors->size() != points->size()) {
			cerr << "Color array missing or size mismatch: colors="
				<< (colors ? colors->size() : 0)
				<< " points=" << points->size() << endl;
			return;
		}
		const bool hasNormals = (normals.valid() && normals->size() == points->size());
		const bool hasSF = (kfIdSF.valid() && kfIdSF->size() == points->size());

		try {
			// ply header
			outFile << "ply\n";
			outFile << "format binary_little_endian 1.0\n";  // 使用二进制小端格式
			outFile << "element vertex " << points->size() << "\n";

			outFile << "property float x\n";
			outFile << "property float y\n";
			outFile << "property float z\n";

			outFile << "property uchar red\n";
			outFile << "property uchar green\n";
			outFile << "property uchar blue\n";

			if (hasNormals) {
				outFile << "property float nx\n";
				outFile << "property float ny\n";
				outFile << "property float nz\n";
			}
			if (hasSF) {
				outFile << "property float scalar_field\n";
			}
			outFile << "end_header\n";

			for (size_t i = 0; i < points->size(); ++i)
			{
				osg::Vec3& p = (*points)[i];
				float x = p.x(), y = p.y(), z = p.z();
				outFile.write(reinterpret_cast<const char*>(&x), sizeof(float));
				outFile.write(reinterpret_cast<const char*>(&y), sizeof(float));
				outFile.write(reinterpret_cast<const char*>(&z), sizeof(float));

				osg::Vec3ub& c = (*colors)[i];
				unsigned char r = c.r(), g = c.g(), b = c.b();
				outFile.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
				outFile.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
				outFile.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));

				if (hasNormals) {
					const osg::Vec3& n = (*normals)[i];
					float nx = n.x(), ny = n.y(), nz = n.z();
					outFile.write(reinterpret_cast<const char*>(&nx), sizeof(float));
					outFile.write(reinterpret_cast<const char*>(&ny), sizeof(float));
					outFile.write(reinterpret_cast<const char*>(&nz), sizeof(float));
				}
				if (hasSF) {
					float kf_id = static_cast<float>((*kfIdSF)[i]);
					outFile.write(reinterpret_cast<const char*>(&kf_id), sizeof(float));
				}
			}
			outFile.close();
			cout << "Point cloud saved to " << filename
				<< (hasSF ? " (with scalar field)" : " (no scalar field)")
				<< (hasNormals ? " (with normals)" : " (no normals)")
				<< endl;
		}
		catch (runtime_error& InternalException) {
			cout << endl
				<< "Exception was thrown: " << InternalException.what()
				<< endl;
		}
	}

	//void PhotoneoSensor::InsertKeyFrame(KeyFrame* pKF)
	//{
	//	unique_lock<mutex> lock(mMutexNewKF);
	//	//cout<<"InsertKeyFrame......."<<endl;
	//	mlNewKeyFrames.push_back(pKF);
	//}

	//bool PhotoneoSensor::CheckNewKeyFrames()
	//{
	//	unique_lock<mutex> lock(mMutexNewKF);
	//	return(!mlNewKeyFrames.empty());
	//}
	bool PhotoneoSensor::checkFinish()
	{
		unique_lock<mutex> lock(mMutexFinish);
		return mbFinishRequested;
	}
	void PhotoneoSensor::RequestFinish()
	{
		unique_lock<mutex> lock(mMutexFinish);
		mbFinishRequested = true;
	}

	void PhotoneoSensor::IDNtoPLY(cv::Mat& I, cv::Mat& D, cv::Mat& N, int frameIndex)
	{
		points->clear();
		colors->clear();
		kfIdSF->clear();
		normals->clear();
		cv::Size imSize = D.size();
		int dSample = 1;
		int ui, vi; // index, which is u(or v) - 1.
		for (int du = 0; du < imSize.width / dSample; ++du)
			for (int dv = 0; dv < imSize.height / dSample; ++dv) {
				ui = du * dSample;
				vi = dv * dSample;

				cv::Point2d& xnyn_un = Raymap.at<cv::Point2d>(vi, ui);
				const float Z = D.at<float>(vi, ui);
				cv::Point3f _p = cv::Point3d(xnyn_un.x, xnyn_un.y, 1.0) * Z / 1e3; // unit: m
				points->push_back(osg::Vec3(_p.x, _p.y, _p.z));

				const uchar _c = I.at<uchar>(vi, ui);
				colors->push_back(osg::Vec3ub(_c, _c, _c));

				const cv::Vec3f _n = N.at<cv::Vec3f>(vi, ui);
				normals->push_back(osg::Vec3(_n[0], _n[1], _n[2]));

				kfIdSF->push_back(static_cast<unsigned int>(frameIndex));
			}
		string path = R"(D:\ins_att\ORB_SLAM3-master\evaluation\DB\02_08_23-11-02_O\calibrated_ply\)" + to_string(frameIndex) + ".ply";
		saveModeltoPly(path);
	}
};
