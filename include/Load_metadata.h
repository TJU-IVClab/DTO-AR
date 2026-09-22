#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

using namespace std;

inline void LM(const string& strAssociationFilename, vector<string>& vstrImageFilenamesRGB,
	vector<string>& vstrImageFilenamesD, vector<string>& vstrImageFilenamesN, vector<double>& vTimestamps)
{
	ifstream fAssociation;
	fAssociation.open(strAssociationFilename.c_str());
	while (!fAssociation.eof())
	{
		string s;
		getline(fAssociation, s);
		if (!s.empty())
		{
			stringstream ss;
			ss << s;
			double t;
			string sRGB, sD, sN;
			ss >> t;
			vTimestamps.push_back(t);
			ss >> sRGB;
			vstrImageFilenamesRGB.push_back(sRGB);
			ss >> t;
			ss >> sD;
			vstrImageFilenamesD.push_back(sD);
			ss >> sN;
			vstrImageFilenamesN.push_back(sN);
		}
	}
}
inline bool LDN(const string& pathToSrc, int i, int j, cv::Mat& imGi, cv::Mat& imDi, cv::Mat& imNi, cv::Mat& imGj, cv::Mat& imDj, cv::Mat& imNj)
{
	// Retrieve paths to images
	vector<string> vstrImageFilenamesRGB;
	vector<string> vstrImageFilenamesD;
	vector<string> vstrImageFilenamesN;
	vector<double> vTimestamps;
	LM(pathToSrc + "/association.txt", vstrImageFilenamesRGB, vstrImageFilenamesD, vstrImageFilenamesN, vTimestamps);

	// Check consistency in the number of images and depthmaps
	int nImages = vstrImageFilenamesRGB.size();
	if (vstrImageFilenamesRGB.empty())
	{
		cerr << endl << "No images found in provided path." << endl;
		return false;
	}
	else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
	{
		cerr << endl << "Different number of images for rgb and depth." << endl;
		return false;
	}

	// Vector for tracking time statistics
	vector<float> vTimesTrack;
	vTimesTrack.resize(nImages);

	cout << endl << "-------" << endl;
	cout << "Images in the sequence: " << nImages << endl << endl;

	vector<cv::Mat> imNholder;
	cv::Mat imNx, imNy, imNz;
	string nameN;
	double tframe;
	imGi = cv::imread(pathToSrc + "/" + vstrImageFilenamesRGB[i], cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	imGi.convertTo(imGi, CV_8UC1, 3.0);
	imDi = cv::imread(pathToSrc + "/" + vstrImageFilenamesD[i], cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	//imDi.convertTo(imDi, CV_32FC1, 1e-3);
	nameN = vstrImageFilenamesN[i];
	imNx = cv::imread(pathToSrc + "/" + nameN, cv::IMREAD_UNCHANGED);
	imNy = cv::imread(pathToSrc + "/" + nameN.replace(nameN.find("X"), 1, "Y"), cv::IMREAD_UNCHANGED);
	imNz = cv::imread(pathToSrc + "/" + nameN.replace(nameN.find("Y"), 1, "Z"), cv::IMREAD_UNCHANGED);
	imNholder.push_back(imNx);
	imNholder.push_back(imNy);
	imNholder.push_back(imNz);
	cv::merge(imNholder, imNi);
	tframe = vTimestamps[i];

	imNholder.clear();
	imGj = cv::imread(pathToSrc + "/" + vstrImageFilenamesRGB[j], cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	imGj.convertTo(imGj, CV_8UC1, 3.0);
	imDj = cv::imread(pathToSrc + "/" + vstrImageFilenamesD[j], cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	//imDj.convertTo(imDj, CV_32FC1, 1e-3);
	nameN = vstrImageFilenamesN[j];
	imNx = cv::imread(pathToSrc + "/" + nameN, cv::IMREAD_UNCHANGED);
	imNy = cv::imread(pathToSrc + "/" + nameN.replace(nameN.find("X"), 1, "Y"), cv::IMREAD_UNCHANGED);
	imNz = cv::imread(pathToSrc + "/" + nameN.replace(nameN.find("Y"), 1, "Z"), cv::IMREAD_UNCHANGED);
	imNholder.push_back(imNx);
	imNholder.push_back(imNy);
	imNholder.push_back(imNz);
	cv::merge(imNholder, imNj);
	tframe = vTimestamps[j];
	return true;
}

inline bool LC(const string& pathToSrc, int idx, cv::Mat& imGi, cv::Mat& imDi)
{
	imGi = cv::imread(pathToSrc + "/" + to_string(idx) + "_IMG_Texture_R.tif", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	imGi.convertTo(imGi, CV_8UC1, 3.0);
	imDi = cv::imread(pathToSrc + "/" + to_string(idx) + "_IMG_DepthMap.tif", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
}