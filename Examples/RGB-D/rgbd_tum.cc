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

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cctype>
#include <limits>
#include <string>
#include <stdexcept>

#include <opencv2/core/core.hpp>

#include <System.h>

using namespace std;

#ifndef DTO_AR_SOURCE_DIR
#define DTO_AR_SOURCE_DIR "."
#endif

void OnlineLoop(int, char**, ORB_SLAM3::System&);
void OfflineLoop(int, char**, ORB_SLAM3::System&);
void LoadImages(const string&, vector<string>&, vector<string>&, vector<string>&, vector<string>&, vector<double>&);
bool longestDigitSeqToInt(const std::string&, int&);

int main(int argc, char** argv)
{
	vector<string> defaultArgs;
	vector<char*> effectiveArgv;
	int effectiveArgc = argc;
	char** effectiveArgvRaw = argv;

	if (argc == 1) // just for local debugging
	{
		const string projectRoot = DTO_AR_SOURCE_DIR;
		defaultArgs = {
			argv[0],
			projectRoot + "/Vocabulary/ORBvoc.txt",
			projectRoot + "/Examples/RGB-D/PhotoneoMotionCam_CTR-095.yaml",
			projectRoot + "/evaluation/DB/02_08_23-11-02_O",
			projectRoot + "/evaluation/DB/02_08_23-11-02_O/association.txt"
		};
		effectiveArgv.reserve(defaultArgs.size());
		for (string& arg : defaultArgs) {
			effectiveArgv.push_back(const_cast<char*>(arg.c_str()));
		}
		effectiveArgc = static_cast<int>(effectiveArgv.size());
		effectiveArgvRaw = effectiveArgv.data();
	}
	else if (argc != 3 && argc != 5)
	{
		cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings" << endl;
		cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
		return 1;
	}

	// Create SLAM system. It initializes all system threads and gets ready to process frames. 
	ORB_SLAM3::System SLAM(effectiveArgvRaw[1], effectiveArgvRaw[2], ORB_SLAM3::System::RGBD, true);
	if (SLAM.mbOnline) {
		OnlineLoop(effectiveArgc, effectiveArgvRaw, SLAM);
	}
	else {
		if (effectiveArgc != 5) {
			cerr << endl << "Offline processing requires: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
			return 1;
		}
		OfflineLoop(effectiveArgc, effectiveArgvRaw, SLAM);
	}

	cout << "System has finished, waiting to close the command line..." << endl;
	system("pause");
	return 0;
}

void OnlineLoop(int argc, char** argv, ORB_SLAM3::System& SLAM)
{
	// Main loop
	vector<double> vTimestamps;
	chrono::system_clock::time_point t1, t2;
	double ttrack;
	while (!SLAM.isShutDown()) {
		t1 = chrono::system_clock::now();

		if (!SLAM.SystemPreprocessFrame()) continue;

		// Pass the image to the SLAM system
		SLAM.TrackRGBD(); // rewrite

		t2 = chrono::system_clock::now();
		ttrack = chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
		vTimestamps.push_back(ttrack);
	}

	SLAM.StopReceivingFrame();

	double totalTime = 0;
	for (double t : vTimestamps) totalTime += t;
	cout << "-------" << endl << endl;
	cout << "mean tracking time: " << totalTime / vTimestamps.size() * 1000 << "ms" << endl;
}

void OfflineLoop(int argc, char** argv, ORB_SLAM3::System& SLAM)
{
	// Retrieve paths to images
	vector<string> vstrImageFilenamesRGB;
	vector<string> vstrImageFilenamesD;
	vector<string> vstrImageFilenamesN;
	vector<string> vstrImageFilenamesC;
	vector<double> vTimestamps;
	string strAssociationFilename = string(argv[4]);
	LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vstrImageFilenamesN, vstrImageFilenamesC, vTimestamps);

	// Check consistency in the number of images and depthmaps
	int nImages = vstrImageFilenamesRGB.size();
	if (vstrImageFilenamesRGB.empty())
	{
		cerr << endl << "No images found in provided path." << endl;
		return;
	}
	else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
	{
		cerr << endl << "Different number of images for rgb and depth." << endl;
		return;
	}

	// Vector for tracking time statistics
	vector<float> vTimesTrack;
	vTimesTrack.resize(nImages);

	cout << endl << "-------" << endl;
	cout << "Start processing sequence ..." << endl;
	cout << "Images in the sequence: " << nImages << endl << endl;

	// Main loop
	cv::Mat imRGB, imD, imN, imC;
	cv::Mat imNx, imNy, imNz;
	for (int ni = 0; ni < nImages; ni++)
	{
		// Read image and depthmap from file
		imRGB = cv::imread(string(argv[3]) + "/i/" + vstrImageFilenamesRGB[ni], cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
		imD = cv::imread(string(argv[3]) + "/d/" + vstrImageFilenamesD[ni], cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
		string nameN = vstrImageFilenamesN[ni];
		imNx = cv::imread(string(argv[3]) + "/n/" + nameN, cv::IMREAD_UNCHANGED);
		imNy = cv::imread(string(argv[3]) + "/n/" + nameN.replace(nameN.find("X"), 1, "Y"), cv::IMREAD_UNCHANGED);
		imNz = cv::imread(string(argv[3]) + "/n/" + nameN.replace(nameN.find("Y"), 1, "Z"), cv::IMREAD_UNCHANGED);
		imC = cv::imread(string(argv[3]) + "/c/" + vstrImageFilenamesC[ni], cv::IMREAD_UNCHANGED);
		double tframe = vTimestamps[ni];

		if (imRGB.empty())
		{
			cerr << endl << "Failed to load image at: "
				<< string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
			return;
		}

		int h = imRGB.rows;
		int w = imRGB.cols;
		imN = cv::Mat(h, w, CV_32FC3);
		for (int v = 0; v < h; v++)
			for (int u = 0; u < w; u++) {
				cv::Vec3f n(imNx.at<float>(v, u), imNy.at<float>(v, u), imNz.at<float>(v, u));
				imN.at<cv::Vec3f>(v, u) = n;
			}
		//if (ni == 5) {
		//string name = to_string(ni);
		//cv::Mat imN_dis;
		//imN.convertTo(imN_dis, CV_8UC3, 128.0, 128.0);
		//cv::imwrite(name + "_normalMap.png", imN_dis);
		//cv::waitKey(1);
		//cv::Mat imC_dis;
		//imC.convertTo(imC_dis, CV_8UC1, 255.0);
		//cv::imwrite(name + "_confidenceMap.png", imC_dis);
		//cv::waitKey(1);
		//}

#ifdef COMPILEDWITHC11
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
		std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

		// Evaluation
		int corr_index = {};
		longestDigitSeqToInt(vstrImageFilenamesRGB[ni], corr_index);
		//SLAM.Eval_IDNtoPLY(imRGB, imD, imN, corr_index);

		// Pass the image to the SLAM system
		//SLAM.TrackRGBD(imRGB,imD,tframe);
		SLAM.SystemPreprocessFrame_offline(imRGB, imD, imN, imC, tframe, corr_index);
		SLAM.TrackRGBD();

#ifdef COMPILEDWITHC11
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
		std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

		double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

		vTimesTrack[ni] = ttrack;

		// Wait to load the next frame
		double T = 0;
		if (ni < nImages - 1)
			T = vTimestamps[ni + 1] - tframe;
		else if (ni > 0)
			T = tframe - vTimestamps[ni - 1];

		if (ttrack < T) {
			long usec = static_cast<long>((T - ttrack) * 1e6);
			std::this_thread::sleep_for(std::chrono::microseconds(usec));
		}
		cout << "{ni}->" << ni << endl;
	}

	// Stop all threads
	SLAM.Shutdown();
	while (!SLAM.isShutDown());
	SLAM.StopReceivingFrame();

	// Tracking time statistics
	sort(vTimesTrack.begin(), vTimesTrack.end());
	float totaltime = 0;
	for (int ni = 0; ni < nImages; ni++)
	{
		totaltime += vTimesTrack[ni];
		cout << vTimesTrack[ni] << endl;
	}
	cout << "-------" << endl << endl;
	cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
	cout << "mean tracking time: " << totaltime / nImages << endl;

	// Save camera trajectory
	SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
	SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void LoadImages(const string& strAssociationFilename, vector<string>& vstrImageFilenamesRGB,
	vector<string>& vstrImageFilenamesD, vector<string>& vstrImageFilenamesN, vector<string>& vstrImageFilenamesC, vector<double>& vTimestamps)
{
	ifstream fAssociation;
	fAssociation.open(strAssociationFilename.c_str());
	while (!fAssociation.eof())
	{
		string s;
		getline(fAssociation, s);

		// 跳过空行
		if (s.empty()) continue;

		// 跳过前导空白
		size_t pos = 0;
		while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;

		// 空白后仍为空 -> 跳过
		if (pos == s.size()) continue;

		// 若段首（忽略前导空格）以 # 开始 -> 忽略此行
		if (s[pos] == '#') continue;

		stringstream ss;
		ss << s;
		double t;
		string sRGB, sD, sN, sC;
		ss >> t;
		vTimestamps.push_back(t * 3);
		ss >> sRGB;
		vstrImageFilenamesRGB.push_back(sRGB);
		ss >> t;
		ss >> sD;
		vstrImageFilenamesD.push_back(sD);
		ss >> sN;
		vstrImageFilenamesN.push_back(sN);
		ss >> sC;
		vstrImageFilenamesC.push_back(sC);
	}
}

bool longestDigitSeqToInt(const std::string& input, int& out)
{
	std::string s = input;
	std::size_t dot = s.find_last_of('.');
	if (dot != std::string::npos) {
		s.erase(dot);
	}

	std::size_t bestStart = 0, bestLen = 0;
	std::size_t curStart = 0, curLen = 0;

	for (std::size_t i = 0; i < s.size(); ++i) {
		unsigned char ch = static_cast<unsigned char>(s[i]);
		if (std::isdigit(ch)) {
			if (curLen == 0) curStart = i;
			++curLen;
		}
		else {
			if (curLen > bestLen) { // for same length, take left
				bestStart = curStart;
				bestLen = curLen;
			}
			curLen = 0;
		}
	}

	if (curLen > bestLen) { // end with num
		bestStart = curStart;
		bestLen = curLen;
	}

	if (bestLen == 0) return false; // no num

	long long val = 0;
	try {
		val = std::stoll(s.substr(bestStart, bestLen));
	}
	catch (const std::exception&) {
		return false;
	}

	// range check
	if (val < std::numeric_limits<int>::min() || val > std::numeric_limits<int>::max())
		return false;

	out = static_cast<int>(val);
	return true;
}
