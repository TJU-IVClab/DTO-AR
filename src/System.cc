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



#include "System.h"
#include "Converter.h"
// !IVC-lab@lee import for offline GBA
#include "Optimizer.h"
#include "DenseOptimizer.cuh"

#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#if defined(USE_HASHLIBPP)
#include <hl_md5.h>
#else
#include <openssl/md5.h>
#endif

extern "C" void denseOptPoseSE3(
	cv::Mat mK, cv::Mat mDistCoeffs, cv::Mat Raymap,
	vector<shared_ptr<cv::Mat>> mvdepth_KF,
	vector<shared_ptr<cv::Mat>> mvcolor_KF,
	vector<shared_ptr<cv::Mat>> mvnormal_KF,
	vector<Eigen::Matrix4f> mvTwc_KF,
	vector<Eigen::Matrix4f>&mvTwc_KF_opt, int maxIter = 20, float tol = 1e-6);
extern "C" void denseOptPoseSE3PCG(
	cv::Mat mK, cv::Mat mDistCoeffs, cv::Mat Raymap,
	vector<ORB_SLAM3::KeyFrame*> vpKFs, vector<pair<ORB_SLAM3::KeyFrame*, Eigen::Matrix4f>>&mvTwc_KF_opt,
	int maxIter = 20, float tol = 1e-6);

namespace ORB_SLAM3
{

	Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

	System::System(const string& strVocFile, const string& strSettingsFile, const eSensor sensor,
		const bool bUseViewer, const int initFr, const string& strSequence) :
		mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false), mbResetActiveMap(false),
		mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false), mbShutDown(false)
	{
		// Output welcome message
		cout << endl <<
			"ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
			"ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
			"This program comes with ABSOLUTELY NO WARRANTY;" << endl <<
			"This is free software, and you are welcome to redistribute it" << endl <<
			"under certain conditions. See LICENSE.txt." << endl << endl;

		cout << "Input sensor was set to: ";

		if (mSensor == MONOCULAR)
			cout << "Monocular" << endl;
		else if (mSensor == STEREO)
			cout << "Stereo" << endl;
		else if (mSensor == RGBD)
			cout << "RGB-D" << endl;
		else if (mSensor == IMU_MONOCULAR)
			cout << "Monocular-Inertial" << endl;
		else if (mSensor == IMU_STEREO)
			cout << "Stereo-Inertial" << endl;
		else if (mSensor == IMU_RGBD)
			cout << "RGB-D-Inertial" << endl;

		//Check settings file
		cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
		if (!fsSettings.isOpened())
		{
			cerr << "Failed to open settings file at: " << strSettingsFile << endl;
			exit(-1);
		}

		cv::FileNode node = fsSettings["File.version"];
		if (!node.empty() && node.isString() && node.string() == "1.0") {
			settings_ = new Settings(strSettingsFile, mSensor);

			mStrLoadAtlasFromFile = settings_->atlasLoadFile();
			mStrSaveAtlasToFile = settings_->atlasSaveFile();

			cout << (*settings_) << endl;
		}
		else {
			settings_ = nullptr;
			cv::FileNode node = fsSettings["System.LoadAtlasFromFile"];
			if (!node.empty() && node.isString())
			{
				mStrLoadAtlasFromFile = (string)node;
			}

			node = fsSettings["System.SaveAtlasToFile"];
			if (!node.empty() && node.isString())
			{
				mStrSaveAtlasToFile = (string)node;
			}
		}

		node = fsSettings["loopClosing"];
		bool activeLC = true;
		if (!node.empty())
		{
			activeLC = static_cast<int>(fsSettings["loopClosing"]) != 0;
		}

		node = fsSettings["useMatteSpots"]; // !IVC-lab@lee
		mbUseMS = false;
		if (!node.empty())
		{
			mbUseMS = static_cast<int>(fsSettings["useMatteSpots"]) != 0;
		}
		if (!mbUseMS) {

			mStrVocabularyFilePath = strVocFile;

			bool loadedAtlas = false;

			if (mStrLoadAtlasFromFile.empty())
			{
				//Load ORB Vocabulary
				cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

				mpVocabulary = new ORBVocabulary();
				bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
				if (!bVocLoad)
				{
					cerr << "Wrong path to vocabulary. " << endl;
					cerr << "Falied to open at: " << strVocFile << endl;
					exit(-1);
				}
				cout << "Vocabulary loaded!" << endl << endl;

				//Create KeyFrame Database
				mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

				//Create the Atlas
				cout << "Initialization of Atlas from scratch " << endl;
				mpAtlas = new Atlas(0);
			}
			else
			{
				//Load ORB Vocabulary
				cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

				mpVocabulary = new ORBVocabulary();
				bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
				if (!bVocLoad)
				{
					cerr << "Wrong path to vocabulary. " << endl;
					cerr << "Falied to open at: " << strVocFile << endl;
					exit(-1);
				}
				cout << "Vocabulary loaded!" << endl << endl;

				//Create KeyFrame Database
				mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

				cout << "Load File" << endl;

				// Load the file with an earlier session
				//clock_t start = clock();
				cout << "Initialization of Atlas from file: " << mStrLoadAtlasFromFile << endl;
				bool isRead = LoadAtlas(FileType::BINARY_FILE);

				if (!isRead)
				{
					cout << "Error to load the file, please try with other session file or vocabulary file" << endl;
					exit(-1);
				}
				//mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);


				//cout << "KF in DB: " << mpKeyFrameDatabase->mnNumKFs << "; words: " << mpKeyFrameDatabase->mnNumWords << endl;

				loadedAtlas = true;

				mpAtlas->CreateNewMap();

				//clock_t timeElapsed = clock() - start;
				//unsigned msElapsed = timeElapsed / (CLOCKS_PER_SEC / 1000);
				//cout << "Binary file read in " << msElapsed << " ms" << endl;

				//usleep(10*1000*1000);
			}
		}
		else {
			mpVocabulary = new ORBVocabulary();

			//Create KeyFrame Database
			mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

			//Create the Atlas
			mpAtlas = new Atlas(0);
		}

		if (mSensor == IMU_STEREO || mSensor == IMU_MONOCULAR || mSensor == IMU_RGBD)
			mpAtlas->SetInertialSensor();

		//Create Drawers. These are used by the Viewer
		mpFrameDrawer = new FrameDrawer(mpAtlas);
		mpMapDrawer = new MapDrawer(mpAtlas, strSettingsFile, settings_);

		//Initialize the Tracking thread
		//(it will live in the main thread of execution, the one that called this constructor)
		cout << "Seq. Name: " << strSequence << endl;
		mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
			mpAtlas, mpKeyFrameDatabase, strSettingsFile, mSensor, settings_, strSequence);

		//Initialize the Local Mapping thread and launch
		mpLocalMapper = new LocalMapping(this, mpAtlas, mSensor == MONOCULAR || mSensor == IMU_MONOCULAR,
			mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD, strSequence);
		mptLocalMapping = new thread(&ORB_SLAM3::LocalMapping::Run, mpLocalMapper);
		mpLocalMapper->mInitFr = initFr;
		if (settings_)
			mpLocalMapper->mThFarPoints = settings_->thFarPoints();
		else
			mpLocalMapper->mThFarPoints = fsSettings["thFarPoints"];
		if (mpLocalMapper->mThFarPoints != 0)
		{
			cout << "Discard points further than " << mpLocalMapper->mThFarPoints << " m from current camera" << endl;
			mpLocalMapper->mbFarPoints = true;
		}
		else
			mpLocalMapper->mbFarPoints = false;

		//Initialize the Loop Closing thread and launch
		// mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR
		mpLoopCloser = new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR, activeLC); // mSensor!=MONOCULAR);
		mptLoopClosing = new thread(&ORB_SLAM3::LoopClosing::Run, mpLoopCloser);

		// !IVC-lab@lee
		mpScanner = new PhotoneoSensor(settings_); // also the source of offline data
		mptOsgViewer = new thread(&ORB_SLAM3::PhotoneoSensor::renderingPC, mpScanner);
		mbOnline = true;
		node = fsSettings["onlineProcessing"];
		if (!node.empty()) {
			mbOnline = static_cast<int>(fsSettings["onlineProcessing"]) != 0;
		}
		mbOnline ? mpScanner->createFirstConnected() : void();
		// Evaluation
		mbDenseOpt = true;
		node = fsSettings["denseOpt"];
		if (!node.empty()) {
			mbDenseOpt = static_cast<int>(fsSettings["denseOpt"]) != 0;
		}

		mnDenseOptMaxIter = 20;
		node = fsSettings["denseOptMaxIter"];
		if (!node.empty()) {
			mnDenseOptMaxIter = static_cast<int>(node);
		}

		//Set pointers between threads
		mpTracker->SetLocalMapper(mpLocalMapper);
		mpTracker->SetLoopClosing(mpLoopCloser);

		mpLocalMapper->SetTracker(mpTracker);
		mpLocalMapper->SetLoopCloser(mpLoopCloser);

		mpLoopCloser->SetTracker(mpTracker);
		mpLoopCloser->SetLocalMapper(mpLocalMapper);

		//usleep(10*1000*1000);

		//Initialize the Viewer thread and launch
		if (bUseViewer)
			//if(false) // TODO
		{
			mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile, settings_);
			mptViewer = new thread(&Viewer::Run, mpViewer);
			mpTracker->SetViewer(mpViewer);
			mpLoopCloser->mpViewer = mpViewer;
			mpViewer->both = mpFrameDrawer->both;
		}

		// Fix verbosity
		Verbose::SetTh(Verbose::VERBOSITY_QUIET);

	}

	// !IVC-lab@lee
	bool System::SystemPreprocessFrame()
	{
		bool available = mpScanner->getFrame();
		if (available) {
			mpFrameData = mpScanner->mFrame;
		}
		return available;
	}
	void System::SystemPreprocessFrame_offline(cv::Mat& im, cv::Mat& depthmap, cv::Mat& normalmap, cv::Mat& confidencemap, double& timestamp, int& frameIndex)
	{
		mpScanner->getFrame_offline(im, depthmap, normalmap, confidencemap, timestamp, frameIndex);
		mpFrameData = mpScanner->mFrame;
	}
	void System::RenderFrame(KeyFrame* pKF)
	{
		mpScanner->mpcdHandler->setData(
			pKF->frameIndex,
			pKF->GetPoseInverse().matrix(),
			pKF->depth_KF,
			pKF->color_KF
		);
	}
	void System::StopReceivingFrame()
	{
		// Shutdown requests exit; wait before rebuilding the shared point arrays.
		if (mptOsgViewer && mptOsgViewer->joinable())
			mptOsgViewer->join();

		mbOnline ? mpScanner->CorrectDisconnectExample() : void();

		auto now = chrono::system_clock::now();
		time_t now_time_t = chrono::system_clock::to_time_t(now);
		tm now_tm = *localtime(&now_time_t);
		char buffer[100];
		strftime(buffer, sizeof(buffer), "%m_%d_%H-%M", &now_tm);
		string time_str(buffer);
		const string dbPrefix = R"(D:\_projs\ORB_SLAM3-master\evaluation\DB\)" + time_str;
		ensure_dir(dbPrefix);
		if (mbOnline) { // only save raw map online
			ensure_dir(dbPrefix + "/c");
			ensure_dir(dbPrefix + "/d");
			ensure_dir(dbPrefix + "/n");
			ensure_dir(dbPrefix + "/i");
		}

		if (mbUseMS) {
			RunOfflineGBAandUpdate(dbPrefix);
			SaveMStoTxt(dbPrefix + "/spots.txt");
			SaveTrajectoryTUM(dbPrefix + "/CameraTrajectory.txt");
			SaveKeyFrameTrajectoryTUM(dbPrefix + "/KeyFrameTrajectory.txt");
		}
		mpScanner->saveModeltoPly(dbPrefix + "/model.ply");
	}
	void System::RunOfflineGBAandUpdate(string dbPrefix)
	{
		Map* pMap = mpAtlas->GetCurrentMap();
		if (pMap->GetAllKeyFrames().size() < 2) return;

		cout << "Conducting GBA | Progress: 0 / 2" << endl;

		bool mbStopGBA = false;
		unsigned long nLoopKF = pMap->GetOriginKF()->mnId;
		Optimizer::GlobalBundleAdjustemnt(pMap, 10, &mbStopGBA, nLoopKF, true, mbUseMS);

		cout << "Conducting GBA | Progress: 1 / 2" << endl;

		vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

		if (mbOnline) { // only save raw map online
			for (KeyFrame* pKF : vpKFs)
			{
				if (pKF->isBad() || !pKF->hasValidMSTransform) continue;

				try {
					// store raw data for further evaluation
					cv::imwrite((dbPrefix + "/d/" + to_string(pKF->mnId) + ".tif"), pKF->depth_KF);
					//cv::imwrite((dbPrefix + "/n/" + to_string(pKF->mnId) + ".tif"), pKF->normal_KF);
					std::vector<cv::Mat> ch(3);
					cv::split(pKF->normal_KF, ch); // ch[0], ch[1], ch[2]
					if (!cv::imwrite((dbPrefix + "/n/" + to_string(pKF->mnId) + "_X.tif"), ch[0])) std::cerr << "Failed: X path" << "\n";
					if (!cv::imwrite((dbPrefix + "/n/" + to_string(pKF->mnId) + "_Y.tif"), ch[1])) std::cerr << "Failed: Y path" << "\n";
					if (!cv::imwrite((dbPrefix + "/n/" + to_string(pKF->mnId) + "_Z.tif"), ch[2])) std::cerr << "Failed: Z path" << "\n";
					cv::imwrite((dbPrefix + "/c/" + to_string(pKF->mnId) + ".tif"), pKF->confid_KF);
					cv::imwrite((dbPrefix + "/i/" + to_string(pKF->mnId) + ".png"), pKF->color_KF);
				}
				catch (cv::Exception e) {
					cout << e.what() << endl;
				}
			}
			cout << "--------- Keyframes raw data saved! ---------" << endl;
		}

		mpScanner->points = new osg::Vec3Array();
		mpScanner->colors = new osg::Vec3ubArray();
		mpScanner->kfIdSF = new osg::UIntArray();
		mpScanner->normals = new osg::Vec3Array();

		// ablation study
		if (!mbDenseOpt)
		{
			for (KeyFrame* pKF : vpKFs)
			{
				const Eigen::Matrix4f& Twc = pKF->GetPoseInverse().matrix();
				const Eigen::Matrix3f Rwc = Twc.block<3, 3>(0, 0);
				auto& depth = pKF->depth_KF;
				auto& color = pKF->color_KF;
				auto& normal = pKF->normal_KF;
				cv::Size imSize = (depth).size();
				int dSample = 3;
				int ui, vi; // index, which is u(or v) - 1.
				for (int du = 0; du < imSize.width / dSample; ++du)
					for (int dv = 0; dv < imSize.height / dSample; ++dv)
					{
						ui = du * dSample;
						vi = dv * dSample;

						cv::Point2d& xnyn_un = mpScanner->Raymap.at<cv::Point2d>(vi, ui);
						const float& Z = depth.at<float>(vi, ui);
						cv::Point3f _p = cv::Point3d(xnyn_un.x, xnyn_un.y, 1.0) * Z / 1e3; // unit: m
						Eigen::Vector4f p = Eigen::Vector4f(_p.x, _p.y, _p.z, 1);
						Eigen::Vector4f pt = Twc * p;
						mpScanner->points->push_back(osg::Vec3(pt.x(), pt.y(), pt.z()));

						const uchar& _c = color.at<uchar>(vi, ui);
						mpScanner->colors->push_back(osg::Vec3ub(_c, _c, _c));
						
						const cv::Vec3f& _n = normal.at<cv::Vec3f>(vi, ui);
						const Eigen::Vector3f nw = Rwc * Eigen::Vector3f(_n[0], _n[1], _n[2]);
						mpScanner->normals->push_back(osg::Vec3(nw.x(), nw.y(), nw.z()));

						mpScanner->kfIdSF->push_back(static_cast<unsigned int>(pKF->frameIndex));
					}
			}
			cout << "Conducting GBA | Progress: 2 / 2" << endl;
		}
		else {
			// pairwise ICP with PCG
			/*sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
			vector<Eigen::Matrix4f> mvTwc_KF_opt, mvTwc_KF;
			vector<shared_ptr<cv::Mat>> mvdepth_KF;
			vector<shared_ptr<cv::Mat>> mvcolor_KF;
			vector<shared_ptr<cv::Mat>> mvnormal_KF;
			for (KeyFrame* pKF : vpKFs)
			{
				if (pKF->isBad() || !pKF->hasValidMSTransform || pKF->depth_KF.empty() || pKF->color_KF.empty()) continue;

				mvTwc_KF.push_back(pKF->GetPoseInverse().matrix());
				mvdepth_KF.push_back(make_shared<cv::Mat>(pKF->depth_KF));
				mvcolor_KF.push_back(make_shared<cv::Mat>(pKF->color_KF));
				mvnormal_KF.push_back(make_shared<cv::Mat>(pKF->normal_KF));
			}
			denseOptPoseSE3(
				mpScanner->mK, mpScanner->mDistCoeffs, mpScanner->Raymap,
				mvdepth_KF, mvcolor_KF, mvnormal_KF, mvTwc_KF, mvTwc_KF_opt
			);
			cout << "Conducting GBA | Progress: 2 / 2" << endl;
			for (int k = 0; k < mvTwc_KF_opt.size(); ++k)
			{
				const Eigen::Matrix4f& Twc = mvTwc_KF_opt[k];
				auto& depth = mvdepth_KF[k];
				auto& color = mvcolor_KF[k];
				cv::Size imSize = (*depth).size();
				int dSample = 3;
				int ui, vi; // index, which is u(or v) - 1.
				for (int du = 0; du < imSize.width / dSample; ++du)
					for (int dv = 0; dv < imSize.height / dSample; ++dv)
					{
						ui = du * dSample;
						vi = dv * dSample;

						cv::Point2d& xnyn_un = mpScanner->Raymap.at<cv::Point2d>(vi, ui);
						const float& Z = depth->at<float>(vi, ui);
						cv::Point3f _p = cv::Point3d(xnyn_un.x, xnyn_un.y, 1.0) * Z / 1e3; // unit: m
						Eigen::Vector4f p = Eigen::Vector4f(_p.x, _p.y, _p.z, 1);
						Eigen::Vector4f pt = Twc * p;
						mpScanner->points->push_back(osg::Vec3(pt.x(), pt.y(), pt.z()));

						const uchar& _c = color->at<uchar>(vi, ui);
						mpScanner->colors->push_back(osg::Vec3ub(_c, _c, _c));
					}
			}*/

			// full ICP with PCG
			vector<pair<KeyFrame*, Eigen::Matrix4f>> mvTwc_KF_opt;
			denseOptPoseSE3PCG(
				mpScanner->mK, mpScanner->mDistCoeffs, mpScanner->Raymap,
				vpKFs, mvTwc_KF_opt, mnDenseOptMaxIter
			);
			cout << "Conducting GBA | Progress: 2 / 2" << endl;
			ofstream f;
			f.open("OKT.txt");
			f << fixed;
			for (int k = 0; k < mvTwc_KF_opt.size(); ++k)
			{
				const KeyFrame* pKF = mvTwc_KF_opt[k].first;
				const Eigen::Matrix4f& Twc = mvTwc_KF_opt[k].second;
				const Eigen::Matrix3f Rwc = Twc.block<3, 3>(0, 0);
				const cv::Mat& depth = pKF->depth_KF;
				const cv::Mat& color = pKF->color_KF;
				const cv::Mat& normal = pKF->normal_KF;
				cv::Size imSize = depth.size();
				int dSample = 3;
				int ui, vi; // index, which is u(or v) - 1.
				for (int du = 0; du < imSize.width / dSample; ++du)
					for (int dv = 0; dv < imSize.height / dSample; ++dv)
					{
						ui = du * dSample;
						vi = dv * dSample;

						cv::Point2d& xnyn_un = mpScanner->Raymap.at<cv::Point2d>(vi, ui);
						const float& Z = depth.at<float>(vi, ui);
						cv::Point3f _p = cv::Point3d(xnyn_un.x, xnyn_un.y, 1.0) * Z / 1e3; // unit: m
						Eigen::Vector4f p = Eigen::Vector4f(_p.x, _p.y, _p.z, 1);
						Eigen::Vector4f pt = Twc * p;
						mpScanner->points->push_back(osg::Vec3(pt.x(), pt.y(), pt.z()));

						const uchar& _c = color.at<uchar>(vi, ui);
						mpScanner->colors->push_back(osg::Vec3ub(_c, _c, _c));

						const cv::Vec3f& _n = normal.at<cv::Vec3f>(vi, ui);
						const Eigen::Vector3f nw = Rwc * Eigen::Vector3f(_n[0], _n[1], _n[2]);
						mpScanner->normals->push_back(osg::Vec3(nw.x(), nw.y(), nw.z()));

						mpScanner->kfIdSF->push_back(static_cast<unsigned int>(pKF->frameIndex));
					}

				// save optimized trajactory for evaluation
				Eigen::Matrix3f R = Twc.block<3, 3>(0, 0);
				Eigen::Vector3f t = Twc.block<3, 1>(0, 3);
				Eigen::Quaternionf q(R);
				f << setprecision(6) << pKF->mTimeStamp << setprecision(9) << " " << t(0) << " " << t(1) << " " << t(2)
					<< " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
			}
			f.close();
		}
		cout << "GBA finished!" << endl;
	}
	void System::SaveMStoTxt(string filePath)
	{
		Map* pCurrentMap = mpAtlas->GetCurrentMap();
		vector<MapPoint*> set = pCurrentMap->GetAllMapPoints();

		ofstream outFile(filePath);
		if (outFile.is_open()) {
			for (MapPoint* pMP : set) {
				Eigen::Vector3f x3D = pMP->GetWorldPos();
				cv::Mat desc = pMP->GetDescriptor();
				outFile <<
					desc.at<float>(0) << " " <<
					x3D.x() << " " << x3D.y() << " " << x3D.z() << " " <<
					desc.at<float>(1) << " " << desc.at<float>(2) << " " << desc.at<float>(3) <<
					endl;
			}
			cout << "Successfully saved " << set.size() << " matte spots." << endl;
		}
		else {
			cerr << "saveMatteSpotsInFile: Could not open file!" << endl;
		}
		outFile.close();
	}
	void System::Eval_IDNtoPLY(cv::Mat& I, cv::Mat& D, cv::Mat& N, int frameIndex)
	{
		mpScanner->IDNtoPLY(I, D, N, frameIndex);
	}

	Sophus::SE3f System::TrackStereo(const cv::Mat& imLeft, const cv::Mat& imRight, const double& timestamp, const vector<IMU::Point>& vImuMeas, string filename)
	{
		if (mSensor != STEREO && mSensor != IMU_STEREO)
		{
			cerr << "ERROR: you called TrackStereo but input sensor was not set to Stereo nor Stereo-Inertial." << endl;
			exit(-1);
		}

		cv::Mat imLeftToFeed, imRightToFeed;
		if (settings_ && settings_->needToRectify()) {
			cv::Mat M1l = settings_->M1l();
			cv::Mat M2l = settings_->M2l();
			cv::Mat M1r = settings_->M1r();
			cv::Mat M2r = settings_->M2r();

			cv::remap(imLeft, imLeftToFeed, M1l, M2l, cv::INTER_LINEAR);
			cv::remap(imRight, imRightToFeed, M1r, M2r, cv::INTER_LINEAR);
		}
		else if (settings_ && settings_->needToResize()) {
			cv::resize(imLeft, imLeftToFeed, settings_->newImSize());
			cv::resize(imRight, imRightToFeed, settings_->newImSize());
		}
		else {
			imLeftToFeed = imLeft.clone();
			imRightToFeed = imRight.clone();
		}

		// Check mode change
		{
			unique_lock<mutex> lock(mMutexMode);
			if (mbActivateLocalizationMode)
			{
				mpLocalMapper->RequestStop();

				// Wait until Local Mapping has effectively stopped
				while (!mpLocalMapper->isStopped())
				{
					std::this_thread::sleep_for(std::chrono::microseconds(1000));
				}

				mpTracker->InformOnlyTracking(true);
				mbActivateLocalizationMode = false;
			}
			if (mbDeactivateLocalizationMode)
			{
				mpTracker->InformOnlyTracking(false);
				mpLocalMapper->Release();
				mbDeactivateLocalizationMode = false;
			}
		}

		// Check reset
		{
			unique_lock<mutex> lock(mMutexReset);
			if (mbReset)
			{
				mpTracker->Reset();
				mbReset = false;
				mbResetActiveMap = false;
			}
			else if (mbResetActiveMap)
			{
				mpTracker->ResetActiveMap();
				mbResetActiveMap = false;
			}
		}

		if (mSensor == System::IMU_STEREO)
			for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
				mpTracker->GrabImuData(vImuMeas[i_imu]);

		// std::cout << "start GrabImageStereo" << std::endl;
		Sophus::SE3f Tcw = mpTracker->GrabImageStereo(imLeftToFeed, imRightToFeed, timestamp, filename);

		// std::cout << "out grabber" << std::endl;

		unique_lock<mutex> lock2(mMutexState);
		mTrackingState = mpTracker->mState;
		mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
		mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

		return Tcw;
	}

	Sophus::SE3f System::TrackRGBD(const cv::Mat& im, const cv::Mat& depthmap, const double& timestamp, const vector<IMU::Point>& vImuMeas, string filename)
	{
		if (mSensor != RGBD && mSensor != IMU_RGBD)
		{
			cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
			exit(-1);
		}

		cv::Mat imToFeed = im.clone();
		cv::Mat imDepthToFeed = depthmap.clone();
		if (settings_ && settings_->needToResize()) {
			cv::Mat resizedIm;
			cv::resize(im, resizedIm, settings_->newImSize());
			imToFeed = resizedIm;

			cv::resize(depthmap, imDepthToFeed, settings_->newImSize());
		}

		// Check mode change
		{
			unique_lock<mutex> lock(mMutexMode);
			if (mbActivateLocalizationMode)
			{
				mpLocalMapper->RequestStop();

				// Wait until Local Mapping has effectively stopped
				while (!mpLocalMapper->isStopped())
				{
					std::this_thread::sleep_for(std::chrono::microseconds(1000));
				}

				mpTracker->InformOnlyTracking(true);
				mbActivateLocalizationMode = false;
			}
			if (mbDeactivateLocalizationMode)
			{
				mpTracker->InformOnlyTracking(false);
				mpLocalMapper->Release();
				mbDeactivateLocalizationMode = false;
			}
		}

		// Check reset
		{
			unique_lock<mutex> lock(mMutexReset);
			if (mbReset)
			{
				mpTracker->Reset();
				mbReset = false;
				mbResetActiveMap = false;
			}
			else if (mbResetActiveMap)
			{
				mpTracker->ResetActiveMap();
				mbResetActiveMap = false;
			}
		}

		if (mSensor == System::IMU_RGBD)
			for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
				mpTracker->GrabImuData(vImuMeas[i_imu]);

		Sophus::SE3f Tcw = mpTracker->GrabImageRGBD(imToFeed, imDepthToFeed, timestamp, filename);

		unique_lock<mutex> lock2(mMutexState);
		mTrackingState = mpTracker->mState;
		mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
		mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
		return Tcw;
	}
	Sophus::SE3f System::TrackRGBD(string filename) { // !IVC-lab@lee

		// Check mode change
		{
			unique_lock<mutex> lock(mMutexMode);
			if (mbActivateLocalizationMode)
			{
				mpLocalMapper->RequestStop();

				// Wait until Local Mapping has effectively stopped
				while (!mpLocalMapper->isStopped())
				{
					std::this_thread::sleep_for(std::chrono::microseconds(1000));
				}

				mpTracker->InformOnlyTracking(true);
				mbActivateLocalizationMode = false;
			}
			if (mbDeactivateLocalizationMode)
			{
				mpTracker->InformOnlyTracking(false);
				mpLocalMapper->Release();
				mbDeactivateLocalizationMode = false;
			}
		}

		// Check reset
		{
			unique_lock<mutex> lock(mMutexReset);
			if (mbReset)
			{
				mpTracker->Reset();
				mbReset = false;
				mbResetActiveMap = false;
			}
			else if (mbResetActiveMap)
			{
				mpTracker->ResetActiveMap();
				mbResetActiveMap = false;
			}
		}

		Sophus::SE3f Tcw = mpTracker->GrabImageRGBD(mpFrameData, filename);

		unique_lock<mutex> lock2(mMutexState);
		mTrackingState = mpTracker->mState;
		mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
		mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
		return Tcw;
	}

	Sophus::SE3f System::TrackMonocular(const cv::Mat& im, const double& timestamp, const vector<IMU::Point>& vImuMeas, string filename)
	{

		{
			unique_lock<mutex> lock(mMutexReset);
			if (mbShutDown)
				return Sophus::SE3f();
		}

		if (mSensor != MONOCULAR && mSensor != IMU_MONOCULAR)
		{
			cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular nor Monocular-Inertial." << endl;
			exit(-1);
		}

		cv::Mat imToFeed = im.clone();
		if (settings_ && settings_->needToResize()) {
			cv::Mat resizedIm;
			cv::resize(im, resizedIm, settings_->newImSize());
			imToFeed = resizedIm;
		}

		// Check mode change
		{
			unique_lock<mutex> lock(mMutexMode);
			if (mbActivateLocalizationMode)
			{
				mpLocalMapper->RequestStop();

				// Wait until Local Mapping has effectively stopped
				while (!mpLocalMapper->isStopped())
				{
					std::this_thread::sleep_for(std::chrono::microseconds(1000));
				}

				mpTracker->InformOnlyTracking(true);
				mbActivateLocalizationMode = false;
			}
			if (mbDeactivateLocalizationMode)
			{
				mpTracker->InformOnlyTracking(false);
				mpLocalMapper->Release();
				mbDeactivateLocalizationMode = false;
			}
		}

		// Check reset
		{
			unique_lock<mutex> lock(mMutexReset);
			if (mbReset)
			{
				mpTracker->Reset();
				mbReset = false;
				mbResetActiveMap = false;
			}
			else if (mbResetActiveMap)
			{
				cout << "SYSTEM-> Reseting active map in monocular case" << endl;
				mpTracker->ResetActiveMap();
				mbResetActiveMap = false;
			}
		}

		if (mSensor == System::IMU_MONOCULAR)
			for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
				mpTracker->GrabImuData(vImuMeas[i_imu]);

		Sophus::SE3f Tcw = mpTracker->GrabImageMonocular(imToFeed, timestamp, filename);

		unique_lock<mutex> lock2(mMutexState);
		mTrackingState = mpTracker->mState;
		mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
		mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

		return Tcw;
	}



	void System::ActivateLocalizationMode()
	{
		unique_lock<mutex> lock(mMutexMode);
		mbActivateLocalizationMode = true;
	}

	void System::DeactivateLocalizationMode()
	{
		unique_lock<mutex> lock(mMutexMode);
		mbDeactivateLocalizationMode = true;
	}

	bool System::MapChanged()
	{
		static int n = 0;
		int curn = mpAtlas->GetLastBigChangeIdx();
		if (n < curn)
		{
			n = curn;
			return true;
		}
		else
			return false;
	}

	void System::Reset()
	{
		unique_lock<mutex> lock(mMutexReset);
		mbReset = true;
	}

	void System::ResetActiveMap()
	{
		unique_lock<mutex> lock(mMutexReset);
		mbResetActiveMap = true;
	}

	void System::Shutdown()
	{
		{
			unique_lock<mutex> lock(mMutexReset);
			mbShutDown = true;
		}

		cout << "Shutdown" << endl;

		mpScanner->RequestFinish(); // !IVC-lab@lee

		mpLocalMapper->RequestFinish();
		mpLoopCloser->RequestFinish();
		if (mpViewer) mpViewer->RequestFinish();

		// Wait until all thread have effectively stopped
		while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
		{
			if (!mpLocalMapper->isFinished())
				cout << "mpLocalMapper is not finished" << endl;
			if (!mpLoopCloser->isFinished())
				cout << "mpLoopCloser is not finished" << endl;
			if (mpLoopCloser->isRunningGBA()) {
				cout << "mpLoopCloser is running GBA" << endl;
				cout << "break anyway..." << endl;
				break;
			}
			std::this_thread::sleep_for(std::chrono::microseconds(5000));
		}

		if (!mStrSaveAtlasToFile.empty())
		{
			Verbose::PrintMess("Atlas saving to file " + mStrSaveAtlasToFile, Verbose::VERBOSITY_NORMAL);
			SaveAtlas(FileType::BINARY_FILE);
		}

		/*if (mpViewer)
		{
			mpViewer->RequestFinish();
			while (!mpViewer->isFinished()) {
				std::this_thread::sleep_for(std::chrono::microseconds(5000));
			}
		}*/
		/*if(mpViewer)
			pangolin::BindToContext("ORB-SLAM2: Map Viewer");*/

#ifdef REGISTER_TIMES
		mpTracker->PrintTimeStats();
#endif


	}

	bool System::isShutDown() {
		unique_lock<mutex> lock(mMutexReset);
		return mbShutDown;
	}

	void System::SaveTrajectoryTUM(const string& filename)
	{
		cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
		if (mSensor == MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
			return;
		}

		vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		Sophus::SE3f Two = vpKFs[0]->GetPoseInverse();

		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
		// which is true when tracking failed (lbL).
		list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
		list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
		list<bool>::iterator lbL = mpTracker->mlbLost.begin();
		for (list<Sophus::SE3f>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
			lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++)
		{
			if (*lbL)
				continue;

			KeyFrame* pKF = *lRit;

			Sophus::SE3f Trw;

			// If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
			while (pKF->isBad())
			{
				Trw = Trw * pKF->mTcp;
				pKF = pKF->GetParent();
			}

			Trw = Trw * pKF->GetPose() * Two;

			Sophus::SE3f Tcw = (*lit) * Trw;
			Sophus::SE3f Twc = Tcw.inverse();

			Eigen::Vector3f twc = Twc.translation();
			Eigen::Quaternionf q = Twc.unit_quaternion();

			f << pKF->frameIndex << " " << setprecision(6) << *lT << " " << setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
		}
		f.close();
		// cout << endl << "trajectory saved!" << endl;
	}

	void System::SaveKeyFrameTrajectoryTUM(const string& filename)
	{
		cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

		vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		for (size_t i = 0; i < vpKFs.size(); i++)
		{
			KeyFrame* pKF = vpKFs[i];

			// pKF->SetPose(pKF->GetPose()*Two);

			if (pKF->isBad())
				continue;

			Sophus::SE3f Twc = pKF->GetPoseInverse();
			Eigen::Quaternionf q = Twc.unit_quaternion();
			Eigen::Vector3f t = Twc.translation();
			f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t(0) << " " << t(1) << " " << t(2)
				<< " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

		}

		f.close();
	}

	void System::SaveTrajectoryEuRoC(const string& filename)
	{

		cout << endl << "Saving trajectory to " << filename << " ..." << endl;
		/*if(mSensor==MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
			return;
		}*/

		vector<Map*> vpMaps = mpAtlas->GetAllMaps();
		int numMaxKFs = 0;
		Map* pBiggerMap;
		std::cout << "There are " << std::to_string(vpMaps.size()) << " maps in the atlas" << std::endl;
		for (Map* pMap : vpMaps)
		{
			std::cout << "  Map " << std::to_string(pMap->GetId()) << " has " << std::to_string(pMap->GetAllKeyFrames().size()) << " KFs" << std::endl;
			if (pMap->GetAllKeyFrames().size() > numMaxKFs)
			{
				numMaxKFs = pMap->GetAllKeyFrames().size();
				pBiggerMap = pMap;
			}
		}

		vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		Sophus::SE3f Twb; // Can be word to cam0 or world to b depending on IMU or not.
		if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD)
			Twb = vpKFs[0]->GetImuPose();
		else
			Twb = vpKFs[0]->GetPoseInverse();

		ofstream f;
		f.open(filename.c_str());
		// cout << "file open" << endl;
		f << fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
		// which is true when tracking failed (lbL).
		list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
		list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
		list<bool>::iterator lbL = mpTracker->mlbLost.begin();

		//cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
		//cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
		//cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
		//cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


		for (auto lit = mpTracker->mlRelativeFramePoses.begin(),
			lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++)
		{
			//cout << "1" << endl;
			if (*lbL)
				continue;


			KeyFrame* pKF = *lRit;
			//cout << "KF: " << pKF->mnId << endl;

			Sophus::SE3f Trw;

			// If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
			if (!pKF)
				continue;

			//cout << "2.5" << endl;

			while (pKF->isBad())
			{
				//cout << " 2.bad" << endl;
				Trw = Trw * pKF->mTcp;
				pKF = pKF->GetParent();
				//cout << "--Parent KF: " << pKF->mnId << endl;
			}

			if (!pKF || pKF->GetMap() != pBiggerMap)
			{
				//cout << "--Parent KF is from another map" << endl;
				continue;
			}

			//cout << "3" << endl;

			Trw = Trw * pKF->GetPose() * Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

			// cout << "4" << endl;

			if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD)
			{
				Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
				Eigen::Quaternionf q = Twb.unit_quaternion();
				Eigen::Vector3f twb = Twb.translation();
				f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
			}
			else
			{
				Sophus::SE3f Twc = ((*lit) * Trw).inverse();
				Eigen::Quaternionf q = Twc.unit_quaternion();
				Eigen::Vector3f twc = Twc.translation();
				f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
			}

			// cout << "5" << endl;
		}
		//cout << "end saving trajectory" << endl;
		f.close();
		cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
	}

	void System::SaveTrajectoryEuRoC(const string& filename, Map* pMap)
	{

		cout << endl << "Saving trajectory of map " << pMap->GetId() << " to " << filename << " ..." << endl;
		/*if(mSensor==MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
			return;
		}*/

		int numMaxKFs = 0;

		vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
		if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD)
			Twb = vpKFs[0]->GetImuPose();
		else
			Twb = vpKFs[0]->GetPoseInverse();

		ofstream f;
		f.open(filename.c_str());
		// cout << "file open" << endl;
		f << fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
		// which is true when tracking failed (lbL).
		list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
		list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
		list<bool>::iterator lbL = mpTracker->mlbLost.begin();

		//cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
		//cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
		//cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
		//cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


		for (auto lit = mpTracker->mlRelativeFramePoses.begin(),
			lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++)
		{
			//cout << "1" << endl;
			if (*lbL)
				continue;


			KeyFrame* pKF = *lRit;
			//cout << "KF: " << pKF->mnId << endl;

			Sophus::SE3f Trw;

			// If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
			if (!pKF)
				continue;

			//cout << "2.5" << endl;

			while (pKF->isBad())
			{
				//cout << " 2.bad" << endl;
				Trw = Trw * pKF->mTcp;
				pKF = pKF->GetParent();
				//cout << "--Parent KF: " << pKF->mnId << endl;
			}

			if (!pKF || pKF->GetMap() != pMap)
			{
				//cout << "--Parent KF is from another map" << endl;
				continue;
			}

			//cout << "3" << endl;

			Trw = Trw * pKF->GetPose() * Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

			// cout << "4" << endl;

			if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD)
			{
				Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
				Eigen::Quaternionf q = Twb.unit_quaternion();
				Eigen::Vector3f twb = Twb.translation();
				f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
			}
			else
			{
				Sophus::SE3f Twc = ((*lit) * Trw).inverse();
				Eigen::Quaternionf q = Twc.unit_quaternion();
				Eigen::Vector3f twc = Twc.translation();
				f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
			}

			// cout << "5" << endl;
		}
		//cout << "end saving trajectory" << endl;
		f.close();
		cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
	}

	/*void System::SaveTrajectoryEuRoC(const string &filename)
	{

		cout << endl << "Saving trajectory to " << filename << " ..." << endl;
		if(mSensor==MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
			return;
		}

		vector<Map*> vpMaps = mpAtlas->GetAllMaps();
		Map* pBiggerMap;
		int numMaxKFs = 0;
		for(Map* pMap :vpMaps)
		{
			if(pMap->GetAllKeyFrames().size() > numMaxKFs)
			{
				numMaxKFs = pMap->GetAllKeyFrames().size();
				pBiggerMap = pMap;
			}
		}

		vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
		sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
		if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
			Twb = vpKFs[0]->GetImuPose_();
		else
			Twb = vpKFs[0]->GetPoseInverse_();

		ofstream f;
		f.open(filename.c_str());
		// cout << "file open" << endl;
		f << fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
		// which is true when tracking failed (lbL).
		list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
		list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
		list<bool>::iterator lbL = mpTracker->mlbLost.begin();

		//cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
		//cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
		//cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
		//cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


		for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
			lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
		{
			//cout << "1" << endl;
			if(*lbL)
				continue;


			KeyFrame* pKF = *lRit;
			//cout << "KF: " << pKF->mnId << endl;

			Sophus::SE3f Trw;

			// If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
			if (!pKF)
				continue;

			//cout << "2.5" << endl;

			while(pKF->isBad())
			{
				//cout << " 2.bad" << endl;
				Trw = Trw * pKF->mTcp;
				pKF = pKF->GetParent();
				//cout << "--Parent KF: " << pKF->mnId << endl;
			}

			if(!pKF || pKF->GetMap() != pBiggerMap)
			{
				//cout << "--Parent KF is from another map" << endl;
				continue;
			}

			//cout << "3" << endl;

			Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

			// cout << "4" << endl;


			if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
			{
				Sophus::SE3f Tbw = pKF->mImuCalib.Tbc_ * (*lit) * Trw;
				Sophus::SE3f Twb = Tbw.inverse();

				Eigen::Vector3f twb = Twb.translation();
				Eigen::Quaternionf q = Twb.unit_quaternion();
				f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
			}
			else
			{
				Sophus::SE3f Tcw = (*lit) * Trw;
				Sophus::SE3f Twc = Tcw.inverse();

				Eigen::Vector3f twc = Twc.translation();
				Eigen::Quaternionf q = Twc.unit_quaternion();
				f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
			}

			// cout << "5" << endl;
		}
		//cout << "end saving trajectory" << endl;
		f.close();
		cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
	}*/


	/*void System::SaveKeyFrameTrajectoryEuRoC_old(const string &filename)
	{
		cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

		vector<Map*> vpMaps = mpAtlas->GetAllMaps();
		Map* pBiggerMap;
		int numMaxKFs = 0;
		for(Map* pMap :vpMaps)
		{
			if(pMap->GetAllKeyFrames().size() > numMaxKFs)
			{
				numMaxKFs = pMap->GetAllKeyFrames().size();
				pBiggerMap = pMap;
			}
		}

		vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
		sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		for(size_t i=0; i<vpKFs.size(); i++)
		{
			KeyFrame* pKF = vpKFs[i];

		   // pKF->SetPose(pKF->GetPose()*Two);

			if(pKF->isBad())
				continue;
			if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
			{
				cv::Mat R = pKF->GetImuRotation().t();
				vector<float> q = Converter::toQuaternion(R);
				cv::Mat twb = pKF->GetImuPosition();
				f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb.at<float>(0) << " " << twb.at<float>(1) << " " << twb.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

			}
			else
			{
				cv::Mat R = pKF->GetRotation();
				vector<float> q = Converter::toQuaternion(R);
				cv::Mat t = pKF->GetCameraCenter();
				f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
			}
		}
		f.close();
	}*/

	void System::SaveKeyFrameTrajectoryEuRoC(const string& filename)
	{
		cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

		vector<Map*> vpMaps = mpAtlas->GetAllMaps();
		Map* pBiggerMap;
		int numMaxKFs = 0;
		for (Map* pMap : vpMaps)
		{
			if (pMap && pMap->GetAllKeyFrames().size() > numMaxKFs)
			{
				numMaxKFs = pMap->GetAllKeyFrames().size();
				pBiggerMap = pMap;
			}
		}

		if (!pBiggerMap)
		{
			std::cout << "There is not a map!!" << std::endl;
			return;
		}

		vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		for (size_t i = 0; i < vpKFs.size(); i++)
		{
			KeyFrame* pKF = vpKFs[i];

			// pKF->SetPose(pKF->GetPose()*Two);

			if (!pKF || pKF->isBad())
				continue;
			if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD)
			{
				Sophus::SE3f Twb = pKF->GetImuPose();
				Eigen::Quaternionf q = Twb.unit_quaternion();
				Eigen::Vector3f twb = Twb.translation();
				f << setprecision(6) << 1e9 * pKF->mTimeStamp << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

			}
			else
			{
				Sophus::SE3f Twc = pKF->GetPoseInverse();
				Eigen::Quaternionf q = Twc.unit_quaternion();
				Eigen::Vector3f t = Twc.translation();
				f << setprecision(6) << 1e9 * pKF->mTimeStamp << " " << setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
			}
		}
		f.close();
	}

	void System::SaveKeyFrameTrajectoryEuRoC(const string& filename, Map* pMap)
	{
		cout << endl << "Saving keyframe trajectory of map " << pMap->GetId() << " to " << filename << " ..." << endl;

		vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		for (size_t i = 0; i < vpKFs.size(); i++)
		{
			KeyFrame* pKF = vpKFs[i];

			if (!pKF || pKF->isBad())
				continue;
			if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD)
			{
				Sophus::SE3f Twb = pKF->GetImuPose();
				Eigen::Quaternionf q = Twb.unit_quaternion();
				Eigen::Vector3f twb = Twb.translation();
				f << setprecision(6) << 1e9 * pKF->mTimeStamp << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

			}
			else
			{
				Sophus::SE3f Twc = pKF->GetPoseInverse();
				Eigen::Quaternionf q = Twc.unit_quaternion();
				Eigen::Vector3f t = Twc.translation();
				f << setprecision(6) << 1e9 * pKF->mTimeStamp << " " << setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
			}
		}
		f.close();
	}

	/*void System::SaveTrajectoryKITTI(const string &filename)
	{
		cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
		if(mSensor==MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
			return;
		}

		vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
		sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		cv::Mat Two = vpKFs[0]->GetPoseInverse();

		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
		// which is true when tracking failed (lbL).
		list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
		list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
		for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
		{
			ORB_SLAM3::KeyFrame* pKF = *lRit;

			cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

			while(pKF->isBad())
			{
				Trw = Trw * Converter::toCvMat(pKF->mTcp.matrix());
				pKF = pKF->GetParent();
			}

			Trw = Trw * pKF->GetPoseCv() * Two;

			cv::Mat Tcw = (*lit)*Trw;
			cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
			cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

			f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
				 Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
				 Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
		}
		f.close();
	}*/

	void System::SaveTrajectoryKITTI(const string& filename)
	{
		cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
		if (mSensor == MONOCULAR)
		{
			cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
			return;
		}

		vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
		sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		Sophus::SE3f Tow = vpKFs[0]->GetPoseInverse();

		ofstream f;
		f.open(filename.c_str());
		f << fixed;

		// Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
		// We need to get first the keyframe pose and then concatenate the relative transformation.
		// Frames not localized (tracking failure) are not saved.

		// For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
		// which is true when tracking failed (lbL).
		list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
		list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
		for (list<Sophus::SE3f>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
			lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++)
		{
			ORB_SLAM3::KeyFrame* pKF = *lRit;

			Sophus::SE3f Trw;

			if (!pKF)
				continue;

			while (pKF->isBad())
			{
				Trw = Trw * pKF->mTcp;
				pKF = pKF->GetParent();
			}

			Trw = Trw * pKF->GetPose() * Tow;

			Sophus::SE3f Tcw = (*lit) * Trw;
			Sophus::SE3f Twc = Tcw.inverse();
			Eigen::Matrix3f Rwc = Twc.rotationMatrix();
			Eigen::Vector3f twc = Twc.translation();

			f << setprecision(9) << Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " " << twc(0) << " " <<
				Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " " <<
				Rwc(2, 0) << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << endl;
		}
		f.close();
	}


	void System::SaveDebugData(const int& initIdx)
	{
		// 0. Save initialization trajectory
		SaveTrajectoryEuRoC("init_FrameTrajectoy_" + to_string(mpLocalMapper->mInitSect) + "_" + to_string(initIdx) + ".txt");

		// 1. Save scale
		ofstream f;
		f.open("init_Scale_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
		f << fixed;
		f << mpLocalMapper->mScale << endl;
		f.close();

		// 2. Save gravity direction
		f.open("init_GDir_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
		f << fixed;
		f << mpLocalMapper->mRwg(0, 0) << "," << mpLocalMapper->mRwg(0, 1) << "," << mpLocalMapper->mRwg(0, 2) << endl;
		f << mpLocalMapper->mRwg(1, 0) << "," << mpLocalMapper->mRwg(1, 1) << "," << mpLocalMapper->mRwg(1, 2) << endl;
		f << mpLocalMapper->mRwg(2, 0) << "," << mpLocalMapper->mRwg(2, 1) << "," << mpLocalMapper->mRwg(2, 2) << endl;
		f.close();

		// 3. Save computational cost
		f.open("init_CompCost_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
		f << fixed;
		f << mpLocalMapper->mCostTime << endl;
		f.close();

		// 4. Save biases
		f.open("init_Biases_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
		f << fixed;
		f << mpLocalMapper->mbg(0) << "," << mpLocalMapper->mbg(1) << "," << mpLocalMapper->mbg(2) << endl;
		f << mpLocalMapper->mba(0) << "," << mpLocalMapper->mba(1) << "," << mpLocalMapper->mba(2) << endl;
		f.close();

		// 5. Save covariance matrix
		f.open("init_CovMatrix_" + to_string(mpLocalMapper->mInitSect) + "_" + to_string(initIdx) + ".txt", ios_base::app);
		f << fixed;
		for (int i = 0; i < mpLocalMapper->mcovInertial.rows(); i++)
		{
			for (int j = 0; j < mpLocalMapper->mcovInertial.cols(); j++)
			{
				if (j != 0)
					f << ",";
				f << setprecision(15) << mpLocalMapper->mcovInertial(i, j);
			}
			f << endl;
		}
		f.close();

		// 6. Save initialization time
		f.open("init_Time_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
		f << fixed;
		f << mpLocalMapper->mInitTime << endl;
		f.close();
	}


	int System::GetTrackingState()
	{
		unique_lock<mutex> lock(mMutexState);
		return mTrackingState;
	}

	vector<MapPoint*> System::GetTrackedMapPoints()
	{
		unique_lock<mutex> lock(mMutexState);
		return mTrackedMapPoints;
	}

	vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
	{
		unique_lock<mutex> lock(mMutexState);
		return mTrackedKeyPointsUn;
	}

	double System::GetTimeFromIMUInit()
	{
		double aux = mpLocalMapper->GetCurrKFTime() - mpLocalMapper->mFirstTs;
		if ((aux > 0.) && mpAtlas->isImuInitialized())
			return mpLocalMapper->GetCurrKFTime() - mpLocalMapper->mFirstTs;
		else
			return 0.f;
	}

	bool System::isLost()
	{
		if (!mpAtlas->isImuInitialized())
			return false;
		else
		{
			if ((mpTracker->mState == Tracking::LOST)) //||(mpTracker->mState==Tracking::RECENTLY_LOST))
				return true;
			else
				return false;
		}
	}


	bool System::isFinished()
	{
		return (GetTimeFromIMUInit() > 0.1);
	}

	void System::ChangeDataset()
	{
		if (mpAtlas->GetCurrentMap()->KeyFramesInMap() < 12)
		{
			mpTracker->ResetActiveMap();
		}
		else
		{
			mpTracker->CreateMapInAtlas();
		}

		mpTracker->NewDataset();
	}

	float System::GetImageScale()
	{
		return mpTracker->GetImageScale();
	}

#ifdef REGISTER_TIMES
	void System::InsertRectTime(double& time)
	{
		mpTracker->vdRectStereo_ms.push_back(time);
	}

	void System::InsertResizeTime(double& time)
	{
		mpTracker->vdResizeImage_ms.push_back(time);
	}

	void System::InsertTrackTime(double& time)
	{
		mpTracker->vdTrackTotal_ms.push_back(time);
	}
#endif

	void System::SaveAtlas(int type) {
		if (!mStrSaveAtlasToFile.empty())
		{
			//clock_t start = clock();

			// Save the current session
			mpAtlas->PreSave();

			string pathSaveFileName = "./";
			pathSaveFileName = pathSaveFileName.append(mStrSaveAtlasToFile);
			pathSaveFileName = pathSaveFileName.append(".osa");

			string strVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath, TEXT_FILE);
			std::size_t found = mStrVocabularyFilePath.find_last_of("/\\");
			string strVocabularyName = mStrVocabularyFilePath.substr(found + 1);

			if (type == TEXT_FILE) // File text
			{
				cout << "Starting to write the save text file " << endl;
				std::remove(pathSaveFileName.c_str());
				std::ofstream ofs(pathSaveFileName, std::ios::binary);
				boost::archive::text_oarchive oa(ofs);

				oa << strVocabularyName;
				oa << strVocabularyChecksum;
				oa << mpAtlas;
				cout << "End to write the save text file" << endl;
			}
			else if (type == BINARY_FILE) // File binary
			{
				cout << "Starting to write the save binary file" << endl;
				std::remove(pathSaveFileName.c_str());
				std::ofstream ofs(pathSaveFileName, std::ios::binary);
				boost::archive::binary_oarchive oa(ofs);
				oa << strVocabularyName;
				oa << strVocabularyChecksum;
				oa << mpAtlas;
				cout << "End to write save binary file" << endl;
			}
		}
	}

	bool System::LoadAtlas(int type)
	{
		string strFileVoc, strVocChecksum;
		bool isRead = false;

		string pathLoadFileName = "./";
		pathLoadFileName = pathLoadFileName.append(mStrLoadAtlasFromFile);
		pathLoadFileName = pathLoadFileName.append(".osa");

		if (type == TEXT_FILE) // File text
		{
			cout << "Starting to read the save text file " << endl;
			std::ifstream ifs(pathLoadFileName, std::ios::binary);
			if (!ifs.good())
			{
				cout << "Load file not found" << endl;
				return false;
			}
			boost::archive::text_iarchive ia(ifs);
			ia >> strFileVoc;
			ia >> strVocChecksum;
			ia >> mpAtlas;
			cout << "End to load the save text file " << endl;
			isRead = true;
		}
		else if (type == BINARY_FILE) // File binary
		{
			cout << "Starting to read the save binary file" << endl;
			std::ifstream ifs(pathLoadFileName, std::ios::binary);
			if (!ifs.good())
			{
				cout << "Load file not found" << endl;
				return false;
			}
			boost::archive::binary_iarchive ia(ifs);
			ia >> strFileVoc;
			ia >> strVocChecksum;
			ia >> mpAtlas;
			cout << "End to load the save binary file" << endl;
			isRead = true;
		}

		if (isRead)
		{
			//Check if the vocabulary is the same
			string strInputVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath, TEXT_FILE);

			if (strInputVocabularyChecksum.compare(strVocChecksum) != 0)
			{
				cout << "The vocabulary load isn't the same which the load session was created " << endl;
				cout << "-Vocabulary name: " << strFileVoc << endl;
				return false; // Both are differents
			}

			mpAtlas->SetKeyFrameDababase(mpKeyFrameDatabase);
			mpAtlas->SetORBVocabulary(mpVocabulary);
			mpAtlas->PostLoad();

			return true;
		}
		return false;
	}






	string System::CalculateCheckSum(string filename, int type)
	{
#if defined(USE_HASHLIBPP)
		const int MD5_DIGEST_LENGTH = 16;
#endif

		string checksum = "";

		unsigned char c[MD5_DIGEST_LENGTH];

		std::ios_base::openmode flags = std::ios::in;
		if (type == BINARY_FILE) // Binary file
			flags = std::ios::in | std::ios::binary;

		ifstream f(filename.c_str(), flags);
		if (!f.is_open())
		{
			cout << "[E] Unable to open the in file " << filename << " for Md5 hash." << endl;
			return checksum;
		}

#if defined(USE_HASHLIBPP)
		MD5 md5;
		HL_MD5_CTX md5Context;
		md5.MD5Init(&md5Context);
#else
		MD5_CTX md5Context;
		MD5_Init(&md5Context);
#endif

		char buffer[1024];
		while (int count = f.readsome(buffer, sizeof(buffer)))
		{
#if defined(USE_HASHLIBPP)
			md5.MD5Update(&md5Context, (unsigned char*)(buffer), count);
#else
			MD5_Update(&md5Context, buffer, count);
#endif
		}

		f.close();

#if defined(USE_HASHLIBPP)
		md5.MD5Final(c, &md5Context);
#else
		MD5_Final(c, &md5Context);
#endif
		for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
		{
			char aux[10];
			sprintf(aux, "%02x", c[i]);
			checksum = checksum + aux;
		}

		return checksum;
	}

} //namespace ORB_SLAM

