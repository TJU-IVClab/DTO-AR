/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "RuntimeProfile.h"
#include <cmath>

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>

#include "CameraModels/Pinhole.h"
#include "CameraModels/KannalaBrandt8.h"
#include "SampleAnyTypes.hpp" // !IVC-lab@lee

namespace ORB_SLAM3
{

	long unsigned int Frame::nNextId = 0;
	bool Frame::mbInitialComputations = true;
	float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
	float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
	float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

	//For stereo fisheye matching
	cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

	Frame::Frame() : mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
	{
#ifdef REGISTER_TIMES
		mTimeStereoMatch = 0;
		mTimeORB_Ext = 0;
#endif
	}


	//Copy Constructor
	Frame::Frame(const Frame& frame)
		:mpcpi(frame.mpcpi), mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
		mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
		mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
		mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
		mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
		mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
		mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
		mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
		mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
		mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
		mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
		mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
		mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
		mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
		monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
		mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
		mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
		mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false)
	{
		for (int i = 0; i < FRAME_GRID_COLS; i++)
			for (int j = 0; j < FRAME_GRID_ROWS; j++) {
				mGrid[i][j] = frame.mGrid[i][j];
				if (frame.Nleft > 0) {
					mGridRight[i][j] = frame.mGridRight[i][j];
				}
			}

		if (frame.mbHasPose)
			SetPose(frame.GetPose());

		if (frame.HasVelocity())
		{
			SetVelocity(frame.GetVelocity());
		}

		mmProjectPoints = frame.mmProjectPoints;
		mmMatchedInImage = frame.mmMatchedInImage;

		// !IVC-lab@lee
		mergedCenters = frame.mergedCenters;
		hasValidMSTransform = frame.hasValidMSTransform;

#ifdef REGISTER_TIMES
		mTimeStereoMatch = frame.mTimeStereoMatch;
		mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
	}


	Frame::Frame(const cv::Mat& imLeft, const cv::Mat& imRight, const double& timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat& K, cv::Mat& distCoef, const float& bf, const float& thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib& ImuCalib)
		:mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
		mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
		mpCamera(pCamera), mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
	{
		// Frame ID
		mnId = nNextId++;

		// Scale Level Info
		mnScaleLevels = mpORBextractorLeft->GetLevels();
		mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
		mfLogScaleFactor = log(mfScaleFactor);
		mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
		mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
		mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
		mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

		// ORB extraction
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
		thread threadLeft(&Frame::ExtractORB, this, 0, imLeft, 0, 0);
		thread threadRight(&Frame::ExtractORB, this, 1, imRight, 0, 0);
		threadLeft.join();
		threadRight.join();
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

		mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB).count();
#endif

		N = mvKeys.size();
		if (mvKeys.empty())
			return;

		UndistortKeyPoints();

#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
		ComputeStereoMatches();
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

		mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

		mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));
		mvbOutlier = vector<bool>(N, false);
		mmProjectPoints.clear();
		mmMatchedInImage.clear();


		// This is done only for the first Frame (or after a change in the calibration)
		if (mbInitialComputations)
		{
			ComputeImageBounds(imLeft);

			mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
			mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);



			fx = K.at<float>(0, 0);
			fy = K.at<float>(1, 1);
			cx = K.at<float>(0, 2);
			cy = K.at<float>(1, 2);
			invfx = 1.0f / fx;
			invfy = 1.0f / fy;

			mbInitialComputations = false;
		}

		mb = mbf / fx;

		if (pPrevF)
		{
			if (pPrevF->HasVelocity())
				SetVelocity(pPrevF->GetVelocity());
		}
		else
		{
			mVw.setZero();
		}

		mpMutexImu = new std::mutex();

		//Set no stereo fisheye information
		Nleft = -1;
		Nright = -1;
		mvLeftToRightMatch = vector<int>(0);
		mvRightToLeftMatch = vector<int>(0);
		mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
		monoLeft = -1;
		monoRight = -1;

		AssignFeaturesToGrid();
	}

	Frame::Frame(const cv::Mat& imGray, const cv::Mat& imDepth, const double& timeStamp, ORBextractor* extractor, ORBVocabulary* voc, cv::Mat& K, cv::Mat& distCoef, const float& bf, const float& thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib& ImuCalib)
		:mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
		mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
		mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
		mpCamera(pCamera), mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
	{
		// Frame ID
		mnId = nNextId++;

		// Scale Level Info
		mnScaleLevels = mpORBextractorLeft->GetLevels();
		mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
		mfLogScaleFactor = log(mfScaleFactor);
		mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
		mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
		mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
		mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

		// ORB extraction
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
		ExtractORB(0, imGray, 0, 0);
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

		mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB).count();
#endif


		N = mvKeys.size();

		if (mvKeys.empty())
			return;

		UndistortKeyPoints();

		ComputeStereoFromRGBD(imDepth);

		mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));

		mmProjectPoints.clear();
		mmMatchedInImage.clear();

		mvbOutlier = vector<bool>(N, false);

		// This is done only for the first Frame (or after a change in the calibration)
		if (mbInitialComputations)
		{
			ComputeImageBounds(imGray);

			mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
			mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

			fx = K.at<float>(0, 0);
			fy = K.at<float>(1, 1);
			cx = K.at<float>(0, 2);
			cy = K.at<float>(1, 2);
			invfx = 1.0f / fx;
			invfy = 1.0f / fy;

			mbInitialComputations = false;
		}

		mb = mbf / fx;

		if (pPrevF) {
			if (pPrevF->HasVelocity())
				SetVelocity(pPrevF->GetVelocity());
		}
		else {
			mVw.setZero();
		}

		mpMutexImu = new std::mutex();

		//Set no stereo fisheye information
		Nleft = -1;
		Nright = -1;
		mvLeftToRightMatch = vector<int>(0);
		mvRightToLeftMatch = vector<int>(0);
		mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
		monoLeft = -1;
		monoRight = -1;

		AssignFeaturesToGrid();
	}
	Frame::Frame(const cv::Mat& imGray, const cv::Mat& imDepth, const cv::Mat& imNorm, const cv::Mat& imConf, const double& timeStamp, cv::Mat& K, cv::Mat& distCoef, const float& bf, const float& thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib& ImuCalib)
		:mpcpi(NULL), mpORBvocabulary(nullptr), mpORBextractorLeft(static_cast<ORBextractor*>(NULL)), mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
		mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
		mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
		mpCamera(pCamera), mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false) // !IVC-lab@lee
	{
		// Frame ID
		mnId = nNextId++;

		// Scale Level Info
		mnScaleLevels = 1;
		mfScaleFactor = 1.0f;
		mfLogScaleFactor = log(mfScaleFactor);
		mvScaleFactors = { 1.0f };
		mvInvScaleFactors = { 1.0f };
		mvLevelSigma2 = { 1.0f };
		mvInvLevelSigma2 = { 1.0f };

		// Matte Spots extraction
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
		// Target: register { mvKeys, mvKeysUn, mvDepth, mvuRight, mDescriptors } → mvpMapPoints
		ExtractMatteSpots(imGray, imDepth, imNorm, imConf);

#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

		mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB).count();
#endif

		N = mergedCenters.size();
		if (N < 4) {
			cout << "Frame: Too few matte spots detected!" << endl;
			return;
		}

		SetMSkeysDescripters();

		mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));

		mmProjectPoints.clear();
		mmMatchedInImage.clear();

		mvbOutlier = vector<bool>(N, false);

		// This is done only for the first Frame (or after a change in the calibration)
		if (mbInitialComputations)
		{
			ComputeImageBounds(imGray);

			mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
			mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

			fx = K.at<float>(0, 0);
			fy = K.at<float>(1, 1);
			cx = K.at<float>(0, 2);
			cy = K.at<float>(1, 2);
			invfx = 1.0f / fx;
			invfy = 1.0f / fy;

			mbInitialComputations = false;
		}

		mb = mbf / fx;

		if (pPrevF) {
			if (pPrevF->HasVelocity())
				SetVelocity(pPrevF->GetVelocity());
		}
		else {
			mVw.setZero();
		}

		mpMutexImu = new std::mutex();

		//Set no stereo fisheye information
		Nleft = -1;
		Nright = -1;
		mvLeftToRightMatch = vector<int>(0);
		mvRightToLeftMatch = vector<int>(0);
		mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
		monoLeft = -1;
		monoRight = -1;

		AssignFeaturesToGrid();
	}
	void Frame::ExtractMatteSpots(const cv::Mat& imGray, const cv::Mat& imDepth, const cv::Mat& imNorm, const cv::Mat& imConf)
	{
#if ORB_SLAM3_RUNTIME_PROFILE
		RuntimeProfile::Scope extraction(RuntimeProfile::Extraction);
#endif
#if ORB_SLAM3_RUNTIME_PROFILE && ORB_SLAM3_RUNTIME_PROFILE_DETAIL
		RuntimeProfile::DetailScope detail(false);
#endif
		// Apply Gaussian blur to reduce noise
		cv::Mat blurred;
		cv::GaussianBlur(imGray, blurred, cv::Size(5, 5), 0);

		// Apply Canny edge detection
		cv::Mat edges;
		cv::Canny(blurred, edges, 25, 80, 3, true);

#if ORB_SLAM3_RUNTIME_PROFILE && ORB_SLAM3_RUNTIME_PROFILE_DETAIL
		detail.Next(1);
#endif
		// Ellipse fitting
		vector<vector<cv::Point>> contours;
		vector<cv::Vec4i> hierarchy;
		cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
#if ORB_SLAM3_RUNTIME_PROFILE && ORB_SLAM3_RUNTIME_PROFILE_DETAIL
		detail.Next(2);
#endif
		vector<pair<cv::RotatedRect, cv::RotatedRect>> ellipses;
		for (size_t i = 0; i < contours.size(); i++)
			// At least 10 points are required to fit the ellipse
			if (contours[i].size() >= 15 /*&& hierarchy[i][2] == -1 && hierarchy[i][3] != -1*/)
			{
				vector<cv::Point2f> contours_i, contours_i_un;
				try {
					for (auto& pt : contours[i]) contours_i.push_back({ float(pt.x), float(pt.y) });
					cv::undistortPoints(contours_i, contours_i_un, mK, mDistCoef, cv::Mat(), mK);
				}
				catch (exception E) {
					cout << E.what() << endl;
				}
				cv::RotatedRect e = cv::fitEllipseDirect(contours[i]);
				cv::RotatedRect e_un = cv::fitEllipseDirect(contours_i_un);
				// d(pixel):[5(pi*2.5^2≈20), 21(pi*10.5^2≈350)], MatteSpot{inner⚪} has a true diameter of 10mm
				//float area = (CV_PI * e.size.width * e.size.height / 4.0) / (imGray.size().width == 1680 ? 2.25 : 1.0);
				//float ratio = e.size.width / e.size.height;
				//if (area > 20 && area < 750 && abs(ratio - 1) < 0.5) { // Check validation
				ellipses.push_back({ e, e_un });
				//cout << ellipse.center.x  << "-" << area << endl;
			//}
			}

#if ORB_SLAM3_RUNTIME_PROFILE && ORB_SLAM3_RUNTIME_PROFILE_DETAIL
		detail.Next(3);
#endif
		// Work with concentric ellipses
		float thres_1 = 2.0f; // pixel => Corresponding to the same Matte Spot 🧿
		float thres_2 = 5e-2f; // pixel => Corresponding to the same actual contour 
		for (auto& pair : ellipses)
		{
			const cv::RotatedRect& e = pair.first;
			const cv::RotatedRect& e_un = pair.second;
			bool to_merged = false;
			const float e_area = e.boundingRect().area();

			for (MatteSpot& mergedCenter : mergedCenters)
			{
				cv::RotatedRect& me = mergedCenter.ellipse;
				cv::RotatedRect& me_ot = mergedCenter.ellipse_outer;
				cv::RotatedRect& me_un = mergedCenter.ellipseUn;
				double distance = cv::norm(e.center - me.center);

				if (distance < thres_1)
				{
					const float me_area = me.boundingRect().area();
					const float me_ot_area = me_ot.boundingRect().area();
					to_merged = true;
					// merge concentric ellipses，and take the center of the larger inner ellipse
					if ((2.5f * e_area) < me_area) { me = e; me_un = e_un; }
					else if (e_area > me_area && e_area < 2.5f * me_area) { me = e; me_un = e_un; }
					// take the largest one as the outer ellipse
					if (e_area > me_ot_area) { me_ot = e; }

					mergedCenter.count++;
					break;
				}
			}
			if (!to_merged)
			{
				MatteSpot MS(e, e_un);
				mergedCenters.emplace_back(MS);
			}
		}

#if ORB_SLAM3_RUNTIME_PROFILE && ORB_SLAM3_RUNTIME_PROFILE_DETAIL
		detail.Next(4);
#endif
#if !ORB_SLAM3_ANALYTIC_BOUNDARY_NORMAL
		cv::Mat imGrayUn, imGrayUnF, grad_x, grad_y;
		cv::undistort(imGray, imGrayUn, mK, mDistCoef);
		imGrayUn.convertTo(imGrayUnF, CV_32F, 1.0);
		cv::Scharr(imGrayUnF, grad_x, CV_32F, 1, 0, 3); // dI/dx
		cv::Scharr(imGrayUnF, grad_y, CV_32F, 0, 1, 3); // dI/dy
#endif

		//cv::Mat canvas, canvas_un;
		//cv::cvtColor(imGray, canvas, cv::COLOR_GRAY2RGB);
		//cv::cvtColor(imGrayUn, canvas_un, cv::COLOR_GRAY2RGB);

#if ORB_SLAM3_RUNTIME_PROFILE && ORB_SLAM3_RUNTIME_PROFILE_DETAIL
		detail.Next(5);
#endif
		// Restore the 3D coordinates and normal vectors
#if ORB_SLAM3_DEPTH_CORRECTION
		cv::Mat dep = imDepth.clone();
#endif
		for (auto iter = mergedCenters.begin(); iter != mergedCenters.end();) // Morphological, photometric, depth, and normal verification
		{
			if (iter->count <= 2) {
				iter = mergedCenters.erase(iter); //返回下一个有效的迭代器，无需+1
				continue;
			}
			float ratio = iter->ellipse.size.width / iter->ellipse.size.height;
			float area = (CV_PI * iter->ellipse.size.width * iter->ellipse.size.height / 4.0) / (imGray.size().width == 1680 ? 2.25 : 1.0);
			if (abs(ratio - 1) >= 0.67 || area <= 15 || area >= 350) {
				iter = mergedCenters.erase(iter); //返回下一个有效的迭代器，无需+1
				continue;
			}

			// newly added.
			iter->c = reproject_ideal_to_distorted(iter->ellipseUn.center, mK, mDistCoef);
			//std::cout << iter->ellipse.center << " : " << iter->c << " : " << iter->ellipseUn.center << endl;

			const int uc = static_cast<int>(round(iter->c.x));
			const int vc = static_cast<int>(round(iter->c.y));
			if (uc < 0 || uc >= imGray.size().width || vc < 0 || vc >= imGray.size().height) {
				iter = mergedCenters.erase(iter);
				continue;
			}
			cv::Point2f pts[4]; // _bottomLeft_, _topLeft_, topRight, bottomRight.
			iter->ellipse.points(pts);
			uint8_t Ic = imGray.at<uint8_t>(vc, uc);
			uint8_t Ie = 0;
			bool is_valid = true;
			for (cv::Point2f& pt : pts)
			{
				const int ue = static_cast<int>(round(pt.x));
				const int ve = static_cast<int>(round(pt.y));
				if (ue < 0 || ue >= imGray.size().width || ve < 0 || ve >= imGray.size().height) {
					iter = mergedCenters.erase(iter);
					continue;
				}
				Ie = imGray.at<uint8_t>(ve, ue);
				//cv::circle(canvas, pt, 2, cv::Scalar(255, 255, 0), 1);
				//cout << "!!!" << ptx << " :: " << pty << "::" << int(Ic - Ie) << endl;
				if (Ie == 0) Ie = 1;
				if (Ic * 1.0 / Ie < 1.8) is_valid = false;
			}
			if (!is_valid) { iter = mergedCenters.erase(iter); continue; }

			//correctOneMarker(dep, iter->ellipse, mK, mDistCoef);
#if ORB_SLAM3_DEPTH_CORRECTION
			correctOneMarker_QuadEllipse(dep, iter->ellipse, mK, mDistCoef);
			float sampledZ = sampleBilinear(dep, iter->c);
#else
			float sampledZ = sampleBilinear(imDepth, iter->c);
#endif
			if (sampledZ == 0.0f) { iter = mergedCenters.erase(iter); continue; }
			iter->p.z = sampledZ;
			iter->p.x = (iter->ellipseUn.center.x - mK.at<float>(0, 2)) * sampledZ / mK.at<float>(0, 0);
			iter->p.y = (iter->ellipseUn.center.y - mK.at<float>(1, 2)) * sampledZ / mK.at<float>(1, 1);

			cv::Vec3f sampledN = sampleNormalInEllipse(imNorm, iter->ellipse);
			if (sampledN == cv::Vec3f::zeros()) { iter = mergedCenters.erase(iter); continue; }
			iter->n = cv::Point3d(sampledN[0], sampledN[1], sampledN[2]);

			// For example, a confidence value of 0.12 means that the estimated error for a point measurement is 0.12 mm.
			// This value is based on a heuristic method that considers the light conditions for each pixel.
			iter->confidence = imConf.at<float>(vc, uc);
			if (iter->confidence > 0.60f) { iter = mergedCenters.erase(iter); continue; }

#if ORB_SLAM3_ANALYTIC_BOUNDARY_NORMAL
			const double a = iter->ellipseUn.size.width * 0.5;
			const double b = iter->ellipseUn.size.height * 0.5;
			const double angle = iter->ellipseUn.angle * CV_PI / 180.0;
			if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b)
				|| !std::isfinite(angle) || !std::isfinite(iter->ellipseUn.center.x)
				|| !std::isfinite(iter->ellipseUn.center.y)) {
				iter = mergedCenters.erase(iter);
				continue;
			}
#endif
			iter->subpixelP = gen2dSamplePoints(iter->ellipseUn);
#if ORB_SLAM3_ANALYTIC_BOUNDARY_NORMAL
			const double cosAngle = std::cos(angle), sinAngle = std::sin(angle);
			iter->subpixelG.clear();
			iter->subpixelG.reserve(iter->subpixelP.size());
			for (size_t k = 0; k < iter->subpixelP.size(); ++k) {
				// Same parameter angles as gen2dSamplePoints; normal is not the radius.
				const double t = 2.0 * CV_PI * k / iter->subpixelP.size();
				const double nx = std::cos(t) / a, ny = std::sin(t) / b;
				const double length = std::hypot(nx, ny);
				iter->subpixelG.emplace_back(
					static_cast<float>((cosAngle * nx - sinAngle * ny) / length),
					static_cast<float>((sinAngle * nx + cosAngle * ny) / length));
			}
#else
			for (const cv::Point2f& pt : iter->subpixelP) {
				float gx = sampleBilinear(grad_x, pt);
				float gy = sampleBilinear(grad_y, pt);
				cv::Vec2f n = cv::normalize(cv::Vec2f(gx, gy));
				iter->subpixelG.emplace_back(n);
				// -------------
				//cv::Point2f pt0 = (pt - iter->ellipseUn.center) * 10 + iter->ellipseUn.center;
				//cv::Point2f pt1(
				//	(pt0.x - n[0] * 10.0),
				//	(pt0.y - n[1] * 10.0)
				//);
				//cv::circle(canvas_un, pt0, 1, cv::Scalar(255.0, 0, 0), -1);
				//cv::arrowedLine(canvas_un, pt0, pt1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
				// -------------
			}
#endif
			//cout << " u: " << iter->c.x << ", v: " << iter->c.y << " un: " << iter->ellipseUn.center.x << ", vn: " << iter->ellipseUn.center.y
			//	<< ", x: " << iter->p.x << ", y: " << iter->p.y << ", z: " << iter->p.z
			//	<< ", nx: " << iter->n.x << ", ny: " << iter->n.y << ", nz: " << iter->n.z
			//	<< ", merged: " << iter->count << endl;
			//cv::ellipse(canvas, iter->ellipse, cv::Scalar(0, 255.0, 0), 1);
			//cv::ellipse(canvas, iter->ellipse_outer, cv::Scalar(0, 0, 255.0), 1);

			++iter;
		}
#if ORB_SLAM3_RUNTIME_PROFILE && ORB_SLAM3_RUNTIME_PROFILE_DETAIL
		detail.Next(6);
#endif
		//cv::namedWindow("canvas", cv::WINDOW_NORMAL);
		//cv::resizeWindow("canvas", 1120, 800);
		//cv::imshow("canvas", canvas);
		////cv::imshow("canvas_un", canvas_un);
		//cv::waitKey(0);
	}

	void Frame::SetMSkeysDescripters()
	{
		mvKeys.resize(N);
		mvKeysUn.resize(N);
		mvuRight = vector<float>(N, -1.0f);
		mvDepth = vector<float>(N, -1.0f);

		const int K = 64;
		const int DESC_DIM = 4 + 4 * K; // ID, nx, ny, nz + 32 * (Px, Py, Gx, Gy) = 132
		mDescriptors = cv::Mat(N, DESC_DIM, CV_32F);

		for (int i = 0; i < N; i++)
		{
			const MatteSpot& MS = mergedCenters[i];
			cv::KeyPoint kp, kpUn;
			kp.pt = /*MS.ellipse.center*/MS.c; kp.octave = 0;
			kpUn.pt = MS.ellipseUn.center; kpUn.octave = 0;
			kpUn.response = MS.confidence;
			mvKeys[i] = kp;
			mvKeysUn[i] = kpUn;					// c(u, v)

			const float& d = MS.p.z;
			mvDepth[i] = d;						// p
			mvuRight[i] = kpUn.pt.x - mbf / d;	// stereo

			float* desc = mDescriptors.ptr<float>(i);
			int idx = 0;

			// ID, nx, ny, nz
			desc[idx++] = static_cast<float>(MS.ID);
			desc[idx++] = static_cast<float>(MS.n.x);
			desc[idx++] = static_cast<float>(MS.n.y);
			desc[idx++] = static_cast<float>(MS.n.z);

			// 32 * (Px, Py, Gx, Gy)
			for (int k = 0; k < K; ++k)
			{
				cv::Point2f pt(0.f, 0.f);
				cv::Vec2f   n(0.f, 0.f);

				if (k < MS.subpixelP.size())
					pt = MS.subpixelP[k];
				if (k < MS.subpixelG.size())
					n = MS.subpixelG[k];

				desc[idx++] = pt.x;  // MS.subpixelP[k].x
				desc[idx++] = pt.y;  // MS.subpixelP[k].y
				desc[idx++] = n[0];  // MS.subpixelG[k][0]
				desc[idx++] = n[1];  // MS.subpixelG[k][1]
			}
		}
	}

	void Frame::GetMaskMS(const cv::Size size, cv::Mat& dst)
	{
		// mask：椭圆内为 255
		dst = cv::Mat(size, CV_8UC1, cv::Scalar(0));
		for (const auto& MS : mergedCenters)
		{
			cv::RotatedRect e_ot = MS.ellipse_outer;
			cv::ellipse(dst, e_ot, cv::Scalar(255), cv::FILLED);
		}

		// 边缘生长 grow 像素：膨胀
		const int grow = 15;
		static const cv::Mat se = cv::getStructuringElement(
			cv::MORPH_ELLIPSE,
			cv::Size(2 * grow + 1, 2 * grow + 1)   // 31x31
		);
		const cv::Rect nonzero = cv::boundingRect(dst);
		if (nonzero.empty()) return;
		const cv::Rect expanded(nonzero.x - grow, nonzero.y - grow,
			nonzero.width + 2 * grow, nonzero.height + 2 * grow);
		cv::Mat roi = dst(expanded & cv::Rect(0, 0, dst.cols, dst.rows));
		// All pixels outside this ROI are zero; dilation cannot reach beyond it.
		cv::dilate(roi, roi, se, cv::Point(-1, -1), 1,
			cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::morphologyDefaultBorderValue());
	}
	void Frame::PreserveIntensityInEllipses(const cv::Mat& gray_src, cv::Mat& gray_dst)
	{
		CV_Assert(!gray_src.empty());
		CV_Assert(gray_src.channels() == 1);

		cv::Mat mask;
		GetMaskMS(gray_src.size(), mask);

		gray_dst = cv::Mat::zeros(gray_src.size(), gray_src.type());
		gray_src.copyTo(gray_dst, mask);   // 仅拷贝 mask 内像素，其余保持 0
	}
	void Frame::ZeroDepthInEllipses(const cv::Mat& depth_src, cv::Mat& depth_dst)
	{
		CV_Assert(!depth_src.empty());
		CV_Assert(depth_src.channels() == 1);

		cv::Mat mask;
		GetMaskMS(depth_src.size(), mask);

		depth_src.copyTo(depth_dst);
		depth_dst.setTo(0, mask);
	}
	void Frame::CorrDepthInEllipses(const cv::Mat& depth_src, cv::Mat& depth_dst)
	{
		depth_src.copyTo(depth_dst);
		for (const MatteSpot& MS : mergedCenters)
		{
			//correctOneMarker(depth_dst, MS.ellipse_outer, mK, mDistCoef);
			correctOneMarker_QuadEllipse(depth_dst, MS.ellipse_outer, mK, mDistCoef);
		}
	}

	Frame::Frame(const cv::Mat& imGray, const double& timeStamp, ORBextractor* extractor, ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat& distCoef, const float& bf, const float& thDepth, Frame* pPrevF, const IMU::Calib& ImuCalib)
		:mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
		mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
		mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
		mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
	{
		// Frame ID
		mnId = nNextId++;

		// Scale Level Info
		mnScaleLevels = mpORBextractorLeft->GetLevels();
		mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
		mfLogScaleFactor = log(mfScaleFactor);
		mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
		mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
		mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
		mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

		// ORB extraction
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
		ExtractORB(0, imGray, 0, 1000);
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

		mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB).count();
#endif


		N = mvKeys.size();
		if (mvKeys.empty())
			return;

		UndistortKeyPoints();

		// Set no stereo information
		mvuRight = vector<float>(N, -1);
		mvDepth = vector<float>(N, -1);
		mnCloseMPs = 0;

		mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));

		mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
		mmMatchedInImage.clear();

		mvbOutlier = vector<bool>(N, false);

		// This is done only for the first Frame (or after a change in the calibration)
		if (mbInitialComputations)
		{
			ComputeImageBounds(imGray);

			mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
			mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

			fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0, 0);
			fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1, 1);
			cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0, 2);
			cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1, 2);
			invfx = 1.0f / fx;
			invfy = 1.0f / fy;

			mbInitialComputations = false;
		}


		mb = mbf / fx;

		//Set no stereo fisheye information
		Nleft = -1;
		Nright = -1;
		mvLeftToRightMatch = vector<int>(0);
		mvRightToLeftMatch = vector<int>(0);
		mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
		monoLeft = -1;
		monoRight = -1;

		AssignFeaturesToGrid();

		if (pPrevF)
		{
			if (pPrevF->HasVelocity())
			{
				SetVelocity(pPrevF->GetVelocity());
			}
		}
		else
		{
			mVw.setZero();
		}

		mpMutexImu = new std::mutex();
	}


	void Frame::AssignFeaturesToGrid()
	{
		// Fill matrix with points
		const int nCells = FRAME_GRID_COLS * FRAME_GRID_ROWS;

		int nReserve = 0.5f * N / (nCells);

		for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
			for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
				mGrid[i][j].reserve(nReserve);
				if (Nleft != -1) {
					mGridRight[i][j].reserve(nReserve);
				}
			}



		for (int i = 0; i < N; i++)
		{
			const cv::KeyPoint& kp = (Nleft == -1) ? mvKeysUn[i]
				: (i < Nleft) ? mvKeys[i]
				: mvKeysRight[i - Nleft];

			int nGridPosX, nGridPosY;
			if (PosInGrid(kp, nGridPosX, nGridPosY)) {
				if (Nleft == -1 || i < Nleft)
					mGrid[nGridPosX][nGridPosY].push_back(i);
				else
					mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
			}
		}
	}

	void Frame::ExtractORB(int flag, const cv::Mat& im, const int x0, const int x1)
	{
		vector<int> vLapping = { x0,x1 };
		if (flag == 0)
			monoLeft = (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors, vLapping);
		else
			monoRight = (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight, vLapping);
	}

	bool Frame::isSet() const {
		return mbIsSet;
	}

	void Frame::SetPose(const Sophus::SE3<float>& Tcw) {
		mTcw = Tcw;

		UpdatePoseMatrices();
		mbIsSet = true;
		mbHasPose = true;
	}

	void Frame::SetNewBias(const IMU::Bias& b)
	{
		mImuBias = b;
		if (mpImuPreintegrated)
			mpImuPreintegrated->SetNewBias(b);
	}

	void Frame::SetVelocity(Eigen::Vector3f Vwb)
	{
		mVw = Vwb;
		mbHasVelocity = true;
	}

	Eigen::Vector3f Frame::GetVelocity() const
	{
		return mVw;
	}

	void Frame::SetImuPoseVelocity(const Eigen::Matrix3f& Rwb, const Eigen::Vector3f& twb, const Eigen::Vector3f& Vwb)
	{
		mVw = Vwb;
		mbHasVelocity = true;

		Sophus::SE3f Twb(Rwb, twb);
		Sophus::SE3f Tbw = Twb.inverse();

		mTcw = mImuCalib.mTcb * Tbw;

		UpdatePoseMatrices();
		mbIsSet = true;
		mbHasPose = true;
	}

	void Frame::UpdatePoseMatrices()
	{
		Sophus::SE3<float> Twc = mTcw.inverse();
		mRwc = Twc.rotationMatrix();
		mOw = Twc.translation();
		mRcw = mTcw.rotationMatrix();
		mtcw = mTcw.translation();
	}

	Eigen::Matrix<float, 3, 1> Frame::GetImuPosition() const {
		return mRwc * mImuCalib.mTcb.translation() + mOw;
	}

	Eigen::Matrix<float, 3, 3> Frame::GetImuRotation() {
		return mRwc * mImuCalib.mTcb.rotationMatrix();
	}

	Sophus::SE3<float> Frame::GetImuPose() {
		return mTcw.inverse() * mImuCalib.mTcb;
	}

	Sophus::SE3f Frame::GetRelativePoseTrl()
	{
		return mTrl;
	}

	Sophus::SE3f Frame::GetRelativePoseTlr()
	{
		return mTlr;
	}

	Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation() {
		return mTlr.rotationMatrix();
	}

	Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
		return mTlr.translation();
	}


	bool Frame::isInFrustum(MapPoint* pMP, float viewingCosLimit)
	{
		if (Nleft == -1) {
			pMP->mbTrackInView = false;
			pMP->mTrackProjX = -1;
			pMP->mTrackProjY = -1;

			// 3D in absolute coordinates
			Eigen::Matrix<float, 3, 1> P = pMP->GetWorldPos();

			// 3D in camera coordinates
			const Eigen::Matrix<float, 3, 1> Pc = mRcw * P + mtcw;
			const float Pc_dist = Pc.norm();

			// Check positive depth
			const float& PcZ = Pc(2);
			const float invz = 1.0f / PcZ;
			if (PcZ < 0.0f)
				return false;

			const Eigen::Vector2f uv = mpCamera->project(Pc);

			if (uv(0) < mnMinX || uv(0) > mnMaxX)
				return false;
			if (uv(1) < mnMinY || uv(1) > mnMaxY)
				return false;

			pMP->mTrackProjX = uv(0);
			pMP->mTrackProjY = uv(1);

			// Check distance is in the scale invariance region of the MapPoint
			const float maxDistance = pMP->GetMaxDistanceInvariance();
			const float minDistance = pMP->GetMinDistanceInvariance();
			const Eigen::Vector3f PO = P - mOw;
			const float dist = PO.norm();

			if (dist<minDistance || dist>maxDistance)
				return false;

			// Check viewing angle
			Eigen::Vector3f Pn = pMP->GetNormal();

			const float viewCos = PO.dot(Pn) / dist;

			if (viewCos < viewingCosLimit)
				return false;

			// Predict scale in the image
			const int nPredictedLevel = pMP->PredictScale(dist, this);

			// Data used by the tracking
			pMP->mbTrackInView = true;
			pMP->mTrackProjX = uv(0);
			pMP->mTrackProjXR = uv(0) - mbf * invz;

			pMP->mTrackDepth = Pc_dist;

			pMP->mTrackProjY = uv(1);
			pMP->mnTrackScaleLevel = nPredictedLevel;
			pMP->mTrackViewCos = viewCos;

			return true;
		}
		else {
			pMP->mbTrackInView = false;
			pMP->mbTrackInViewR = false;
			pMP->mnTrackScaleLevel = -1;
			pMP->mnTrackScaleLevelR = -1;

			pMP->mbTrackInView = isInFrustumChecks(pMP, viewingCosLimit);
			pMP->mbTrackInViewR = isInFrustumChecks(pMP, viewingCosLimit, true);

			return pMP->mbTrackInView || pMP->mbTrackInViewR;
		}
	}

	bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f& kp, float& u, float& v)
	{

		// 3D in absolute coordinates
		Eigen::Vector3f P = pMP->GetWorldPos();

		// 3D in camera coordinates
		const Eigen::Vector3f Pc = mRcw * P + mtcw;
		const float& PcX = Pc(0);
		const float& PcY = Pc(1);
		const float& PcZ = Pc(2);

		// Check positive depth
		if (PcZ < 0.0f)
		{
			cout << "Negative depth: " << PcZ << endl;
			return false;
		}

		// Project in image and check it is not outside
		const float invz = 1.0f / PcZ;
		u = fx * PcX * invz + cx;
		v = fy * PcY * invz + cy;

		if (u<mnMinX || u>mnMaxX)
			return false;
		if (v<mnMinY || v>mnMaxY)
			return false;

		float u_distort, v_distort;

		float x = (u - cx) * invfx;
		float y = (v - cy) * invfy;
		float r2 = x * x + y * y;
		float k1 = mDistCoef.at<float>(0);
		float k2 = mDistCoef.at<float>(1);
		float p1 = mDistCoef.at<float>(2);
		float p2 = mDistCoef.at<float>(3);
		float k3 = 0;
		if (mDistCoef.total() == 5)
		{
			k3 = mDistCoef.at<float>(4);
		}

		// Radial distorsion
		float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
		float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

		// Tangential distorsion
		x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
		y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

		u_distort = x_distort * fx + cx;
		v_distort = y_distort * fy + cy;


		u = u_distort;
		v = v_distort;

		kp = cv::Point2f(u, v);

		return true;
	}

	Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
	{
		return mRcw * pCw + mtcw;
	}

	vector<size_t> Frame::GetFeaturesInArea(const float& x, const float& y, const float& r, const int minLevel, const int maxLevel, const bool bRight) const
	{
		vector<size_t> vIndices;
		vIndices.reserve(N);

		float factorX = r;
		float factorY = r;

		const int nMinCellX = max(0, (int)floor((x - mnMinX - factorX) * mfGridElementWidthInv));
		if (nMinCellX >= FRAME_GRID_COLS)
		{
			return vIndices;
		}

		const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + factorX) * mfGridElementWidthInv));
		if (nMaxCellX < 0)
		{
			return vIndices;
		}

		const int nMinCellY = max(0, (int)floor((y - mnMinY - factorY) * mfGridElementHeightInv));
		if (nMinCellY >= FRAME_GRID_ROWS)
		{
			return vIndices;
		}

		const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + factorY) * mfGridElementHeightInv));
		if (nMaxCellY < 0)
		{
			return vIndices;
		}

		const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

		for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
		{
			for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
			{
				const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
				if (vCell.empty())
					continue;

				for (size_t j = 0, jend = vCell.size(); j < jend; j++)
				{
					const cv::KeyPoint& kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
						: (!bRight) ? mvKeys[vCell[j]]
						: mvKeysRight[vCell[j]];
					if (bCheckLevels)
					{
						if (kpUn.octave < minLevel)
							continue;
						if (maxLevel >= 0)
							if (kpUn.octave > maxLevel)
								continue;
					}

					const float distx = kpUn.pt.x - x;
					const float disty = kpUn.pt.y - y;

					if (fabs(distx) < factorX && fabs(disty) < factorY)
						vIndices.push_back(vCell[j]);
				}
			}
		}

		return vIndices;
	}

	bool Frame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY)
	{
		posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
		posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

		//Keypoint's coordinates are undistorted, which could cause to go out of the image
		if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
			return false;

		return true;
	}


	void Frame::ComputeBoW()
	{
		if (mBowVec.empty())
		{
			vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
			mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
		}
	}

	void Frame::UndistortKeyPoints()
	{
		if (mDistCoef.at<float>(0) == 0.0)
		{
			mvKeysUn = mvKeys;
			return;
		}

		// Fill matrix with points
		cv::Mat mat(N, 2, CV_32F);

		for (int i = 0; i < N; i++)
		{
			mat.at<float>(i, 0) = mvKeys[i].pt.x;
			mat.at<float>(i, 1) = mvKeys[i].pt.y;
		}

		// Undistort points
		mat = mat.reshape(2);
		cv::undistortPoints(mat, mat, static_cast<Pinhole*>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
		mat = mat.reshape(1);


		// Fill undistorted keypoint vector
		mvKeysUn.resize(N);
		for (int i = 0; i < N; i++)
		{
			cv::KeyPoint kp = mvKeys[i];
			kp.pt.x = mat.at<float>(i, 0);
			kp.pt.y = mat.at<float>(i, 1);
			mvKeysUn[i] = kp;
		}

	}

	void Frame::ComputeImageBounds(const cv::Mat& imLeft)
	{
		if (mDistCoef.at<float>(0) != 0.0)
		{
			cv::Mat mat(4, 2, CV_32F);
			mat.at<float>(0, 0) = 0.0; mat.at<float>(0, 1) = 0.0;
			mat.at<float>(1, 0) = imLeft.cols; mat.at<float>(1, 1) = 0.0;
			mat.at<float>(2, 0) = 0.0; mat.at<float>(2, 1) = imLeft.rows;
			mat.at<float>(3, 0) = imLeft.cols; mat.at<float>(3, 1) = imLeft.rows;

			mat = mat.reshape(2);
			cv::undistortPoints(mat, mat, static_cast<Pinhole*>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
			mat = mat.reshape(1);

			// Undistort corners
			mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
			mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
			mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
			mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
		}
		else
		{
			mnMinX = 0.0f;
			mnMaxX = imLeft.cols;
			mnMinY = 0.0f;
			mnMaxY = imLeft.rows;
		}
	}

	void Frame::ComputeStereoMatches()
	{
		mvuRight = vector<float>(N, -1.0f);
		mvDepth = vector<float>(N, -1.0f);

		const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

		const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

		//Assign keypoints to row table
		vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());

		for (int i = 0; i < nRows; i++)
			vRowIndices[i].reserve(200);

		const int Nr = mvKeysRight.size();

		for (int iR = 0; iR < Nr; iR++)
		{
			const cv::KeyPoint& kp = mvKeysRight[iR];
			const float& kpY = kp.pt.y;
			const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
			const int maxr = ceil(kpY + r);
			const int minr = floor(kpY - r);

			for (int yi = minr; yi <= maxr; yi++)
				vRowIndices[yi].push_back(iR);
		}

		// Set limits for search
		const float minZ = mb;
		const float minD = 0;
		const float maxD = mbf / minZ;

		// For each left keypoint search a match in the right image
		vector<pair<int, int> > vDistIdx;
		vDistIdx.reserve(N);

		for (int iL = 0; iL < N; iL++)
		{
			const cv::KeyPoint& kpL = mvKeys[iL];
			const int& levelL = kpL.octave;
			const float& vL = kpL.pt.y;
			const float& uL = kpL.pt.x;

			const vector<size_t>& vCandidates = vRowIndices[vL];

			if (vCandidates.empty())
				continue;

			const float minU = uL - maxD;
			const float maxU = uL - minD;

			if (maxU < 0)
				continue;

			int bestDist = ORBmatcher::TH_HIGH;
			size_t bestIdxR = 0;

			const cv::Mat& dL = mDescriptors.row(iL);

			// Compare descriptor to right keypoints
			for (size_t iC = 0; iC < vCandidates.size(); iC++)
			{
				const size_t iR = vCandidates[iC];
				const cv::KeyPoint& kpR = mvKeysRight[iR];

				if (kpR.octave<levelL - 1 || kpR.octave>levelL + 1)
					continue;

				const float& uR = kpR.pt.x;

				if (uR >= minU && uR <= maxU)
				{
					const cv::Mat& dR = mDescriptorsRight.row(iR);
					const int dist = ORBmatcher::DescriptorDistance(dL, dR);

					if (dist < bestDist)
					{
						bestDist = dist;
						bestIdxR = iR;
					}
				}
			}

			// Subpixel match by correlation
			if (bestDist < thOrbDist)
			{
				// coordinates in image pyramid at keypoint scale
				const float uR0 = mvKeysRight[bestIdxR].pt.x;
				const float scaleFactor = mvInvScaleFactors[kpL.octave];
				const float scaleduL = round(kpL.pt.x * scaleFactor);
				const float scaledvL = round(kpL.pt.y * scaleFactor);
				const float scaleduR0 = round(uR0 * scaleFactor);

				// sliding window search
				const int w = 5;
				cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);

				int bestDist = INT_MAX;
				int bestincR = 0;
				const int L = 5;
				vector<float> vDists;
				vDists.resize(2 * L + 1);

				const float iniu = scaleduR0 + L - w;
				const float endu = scaleduR0 + L + w + 1;
				if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
					continue;

				for (int incR = -L; incR <= +L; incR++)
				{
					cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

					float dist = cv::norm(IL, IR, cv::NORM_L1);
					if (dist < bestDist)
					{
						bestDist = dist;
						bestincR = incR;
					}

					vDists[L + incR] = dist;
				}

				if (bestincR == -L || bestincR == L)
					continue;

				// Sub-pixel match (Parabola fitting)
				const float dist1 = vDists[L + bestincR - 1];
				const float dist2 = vDists[L + bestincR];
				const float dist3 = vDists[L + bestincR + 1];

				const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

				if (deltaR < -1 || deltaR>1)
					continue;

				// Re-scaled coordinate
				float bestuR = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

				float disparity = (uL - bestuR);

				if (disparity >= minD && disparity < maxD)
				{
					if (disparity <= 0)
					{
						disparity = 0.01;
						bestuR = uL - 0.01;
					}
					mvDepth[iL] = mbf / disparity;
					mvuRight[iL] = bestuR;
					vDistIdx.push_back(pair<int, int>(bestDist, iL));
				}
			}
		}

		sort(vDistIdx.begin(), vDistIdx.end());
		const float median = vDistIdx[vDistIdx.size() / 2].first;
		const float thDist = 1.5f * 1.4f * median;

		for (int i = vDistIdx.size() - 1; i >= 0; i--)
		{
			if (vDistIdx[i].first < thDist)
				break;
			else
			{
				mvuRight[vDistIdx[i].second] = -1;
				mvDepth[vDistIdx[i].second] = -1;
			}
		}
	}


	void Frame::ComputeStereoFromRGBD(const cv::Mat& imDepth)
	{
		mvuRight = vector<float>(N, -1);
		mvDepth = vector<float>(N, -1);

		for (int i = 0; i < N; i++)
		{
			const cv::KeyPoint& kp = mvKeys[i];
			const cv::KeyPoint& kpU = mvKeysUn[i];

			const float& v = kp.pt.y;
			const float& u = kp.pt.x;

			const float d = imDepth.at<float>(v, u);

			if (d > 0)
			{
				mvDepth[i] = d;
				mvuRight[i] = kpU.pt.x - mbf / d;
			}
		}
	}

	bool Frame::UnprojectStereo(const int& i, Eigen::Vector3f& x3D)
	{
		const float z = mvDepth[i];
		if (z > 0) {
			const float u = mvKeysUn[i].pt.x;
			const float v = mvKeysUn[i].pt.y;
			const float x = (u - cx) * z * invfx;
			const float y = (v - cy) * z * invfy;
			Eigen::Vector3f x3Dc(x, y, z);
			x3D = mRwc * x3Dc + mOw;
			return true;
		}
		else
			return false;
	}

	bool Frame::imuIsPreintegrated()
	{
		unique_lock<std::mutex> lock(*mpMutexImu);
		return mbImuPreintegrated;
	}

	void Frame::setIntegrated()
	{
		unique_lock<std::mutex> lock(*mpMutexImu);
		mbImuPreintegrated = true;
	}

	Frame::Frame(const cv::Mat& imLeft, const cv::Mat& imRight, const double& timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat& K, cv::Mat& distCoef, const float& bf, const float& thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr, Frame* pPrevF, const IMU::Calib& ImuCalib)
		:mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
		mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
		mbHasPose(false), mbHasVelocity(false)

	{
		imgLeft = imLeft.clone();
		imgRight = imRight.clone();

		// Frame ID
		mnId = nNextId++;

		// Scale Level Info
		mnScaleLevels = mpORBextractorLeft->GetLevels();
		mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
		mfLogScaleFactor = log(mfScaleFactor);
		mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
		mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
		mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
		mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

		// ORB extraction
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
		thread threadLeft(&Frame::ExtractORB, this, 0, imLeft, static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0], static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
		thread threadRight(&Frame::ExtractORB, this, 1, imRight, static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0], static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
		threadLeft.join();
		threadRight.join();
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

		mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB).count();
#endif

		Nleft = mvKeys.size();
		Nright = mvKeysRight.size();
		N = Nleft + Nright;

		if (N == 0)
			return;

		// This is done only for the first Frame (or after a change in the calibration)
		if (mbInitialComputations)
		{
			ComputeImageBounds(imLeft);

			mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
			mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

			fx = K.at<float>(0, 0);
			fy = K.at<float>(1, 1);
			cx = K.at<float>(0, 2);
			cy = K.at<float>(1, 2);
			invfx = 1.0f / fx;
			invfy = 1.0f / fy;

			mbInitialComputations = false;
		}

		mb = mbf / fx;

		// Sophus/Eigen
		mTlr = Tlr;
		mTrl = mTlr.inverse();
		mRlr = mTlr.rotationMatrix();
		mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
		ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
		std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

		mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

		//Put all descriptors in the same matrix
		cv::vconcat(mDescriptors, mDescriptorsRight, mDescriptors);

		mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(nullptr));
		mvbOutlier = vector<bool>(N, false);

		AssignFeaturesToGrid();

		mpMutexImu = new std::mutex();

		UndistortKeyPoints();

	}

	void Frame::ComputeStereoFishEyeMatches() {
		//Speed it up by matching keypoints in the lapping area
		vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
		vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

		cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
		cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

		mvLeftToRightMatch = vector<int>(Nleft, -1);
		mvRightToLeftMatch = vector<int>(Nright, -1);
		mvDepth = vector<float>(Nleft, -1.0f);
		mvuRight = vector<float>(Nleft, -1);
		mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
		mnCloseMPs = 0;

		//Perform a brute force between Keypoint in the left and right image
		vector<vector<cv::DMatch>> matches;

		BFmatcher.knnMatch(stereoDescLeft, stereoDescRight, matches, 2);

		int nMatches = 0;
		int descMatches = 0;

		//Check matches using Lowe's ratio
		for (vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it) {
			if ((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7) {
				//For every good match, check parallax and reprojection error to discard spurious matches
				Eigen::Vector3f p3D;
				descMatches++;
				float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
				float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2, mvKeys[(*it)[0].queryIdx + monoLeft], mvKeysRight[(*it)[0].trainIdx + monoRight], mRlr, mtlr, sigma1, sigma2, p3D);
				if (depth > 0.0001f) {
					mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
					mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
					mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
					mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
					nMatches++;
				}
			}
		}
	}

	bool Frame::isInFrustumChecks(MapPoint* pMP, float viewingCosLimit, bool bRight) {
		// 3D in absolute coordinates
		Eigen::Vector3f P = pMP->GetWorldPos();

		Eigen::Matrix3f mR;
		Eigen::Vector3f mt, twc;
		if (bRight) {
			Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
			Eigen::Vector3f trl = mTrl.translation();
			mR = Rrl * mRcw;
			mt = Rrl * mtcw + trl;
			twc = mRwc * mTlr.translation() + mOw;
		}
		else {
			mR = mRcw;
			mt = mtcw;
			twc = mOw;
		}

		// 3D in camera coordinates
		Eigen::Vector3f Pc = mR * P + mt;
		const float Pc_dist = Pc.norm();
		const float& PcZ = Pc(2);

		// Check positive depth
		if (PcZ < 0.0f)
			return false;

		// Project in image and check it is not outside
		Eigen::Vector2f uv;
		if (bRight) uv = mpCamera2->project(Pc);
		else uv = mpCamera->project(Pc);

		if (uv(0) < mnMinX || uv(0) > mnMaxX)
			return false;
		if (uv(1) < mnMinY || uv(1) > mnMaxY)
			return false;

		// Check distance is in the scale invariance region of the MapPoint
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		const Eigen::Vector3f PO = P - twc;
		const float dist = PO.norm();

		if (dist<minDistance || dist>maxDistance)
			return false;

		// Check viewing angle
		Eigen::Vector3f Pn = pMP->GetNormal();

		const float viewCos = PO.dot(Pn) / dist;

		if (viewCos < viewingCosLimit)
			return false;

		// Predict scale in the image
		const int nPredictedLevel = pMP->PredictScale(dist, this);

		if (bRight) {
			pMP->mTrackProjXR = uv(0);
			pMP->mTrackProjYR = uv(1);
			pMP->mnTrackScaleLevelR = nPredictedLevel;
			pMP->mTrackViewCosR = viewCos;
			pMP->mTrackDepthR = Pc_dist;
		}
		else {
			pMP->mTrackProjX = uv(0);
			pMP->mTrackProjY = uv(1);
			pMP->mnTrackScaleLevel = nPredictedLevel;
			pMP->mTrackViewCos = viewCos;
			pMP->mTrackDepth = Pc_dist;
		}

		return true;
	}

	Eigen::Vector3f Frame::UnprojectStereoFishEye(const int& i) {
		return mRwc * mvStereo3Dpoints[i] + mOw;
	}

} //namespace ORB_SLAM
