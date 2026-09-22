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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include"sophus/sim3.hpp"

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"

namespace ORB_SLAM3 // !IVC-lab@lee
{
	struct MatteSpot;
	class MSmatcher { // Singleton
	public:
		static MSmatcher& getInstance() {
			static MSmatcher instance;  // local static variable, ensuring there's only one instance
			return instance;
		}
		struct MatElement {
			double dp;
			double dn; // cos(θ)
			MatElement() : dp(0), dn(1) {}
			MatElement(double dp, double dn) : dp(dp), dn(dn) {}
		};
		void CalculateMPM(vector<vector<MatElement>>& targetMPM, const vector<MatteSpot>& set);
		void CalculateMPM(vector<vector<MatElement>>& targetMPM, const vector<MapPoint*>& set);
		int SetMSCorrespondences(Frame& F, const vector<MapPoint*>& vpMapPoints);
		float GetOverlapRatio(cv::Mat& IDi, cv::Mat& IDr);

		// Search matches between Frame keypoints and projected MapPoints. Returns number of matches
		// Used to track the local map (Tracking)
		int SearchByProjection(Frame& F, const std::vector<MapPoint*>& vpMapPoints, const float th = 1e-4);

		// Project MapPoints tracked in last frame into the current frame and search matches.
		// Used to track from previous frame (Tracking)
		int SearchByProjection(Frame& CurrentFrame, const Frame& LastFrame, const float th, const bool bMono);

		// Project MapPoints seen in KeyFrame into the Frame and search matches.
		// Used in relocalisation (Tracking)
		int SearchByProjection(Frame& CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*>& sAlreadyFound, const float th, const int ORBdist);


		// Project MapPoints into KeyFrame and search for duplicated MapPoints.
		int Fuse(KeyFrame* pKF, const vector<MapPoint*>& vpMapPoints, const float th = 3.0, const bool bRight = false);

		int getNewID() { return ++i_for_ID; }

		// cumulative error evaluation
		//const vector<int> excluded = { 9, 10, 12, 13, 14 };
		//const vector<int> excluded = { 12, 13, 15, 18, 19 };
		//const vector<int> excluded = { 8, 9, 11, 12, 13 };
		const vector<int> excluded = { 0 };
		void AppendExcludedMeasurement(const std::string& text) {
			const std::string output_path = R"(D:\ins_att\ORB_SLAM3-master\evaluation\DB\excluded_measurements.txt)";
			bool needs_newline = false;

			{
				std::ifstream infile(output_path, std::ios::binary);
				if (infile.good()) {
					infile.seekg(0, std::ios::end);
					if (infile.tellg() > 0) {
						infile.seekg(-1, std::ios::end);
						char last_char = '\0';
						infile.get(last_char);
						needs_newline = last_char != '\n';
					}
				}
			}

			std::ofstream outfile(output_path, std::ios::app);
			if (!outfile.is_open()) {
				return;
			}
			if (needs_newline) {
				outfile << '\n';
			}
			outfile << text << '\n';
		}

	public:
		mutex mutex_MS;

	private:
		MSmatcher() {
			std::cout << "Singleton instance created!" << std::endl;
		};
		MSmatcher(const MSmatcher&) = delete;
		MSmatcher& operator=(const MSmatcher&) = delete;

	protected:
		unsigned long i_for_ID = 0; // identical
		//float RadiusByViewingCos(const float& viewCos);
		//bool mbCheckOrientation;
	};
}
namespace ORB_SLAM3
{
	class ORBmatcher
	{
	public:

		ORBmatcher(float nnratio = 0.6, bool checkOri = true);

		// Computes the Hamming distance between two ORB descriptors
		static int DescriptorDistance(const cv::Mat& a, const cv::Mat& b);

		// Search matches between Frame keypoints and projected MapPoints. Returns number of matches
		// Used to track the local map (Tracking)
		int SearchByProjection(Frame& F, const std::vector<MapPoint*>& vpMapPoints, const float th = 3, const bool bFarPoints = false, const float thFarPoints = 50.0f);

		// Project MapPoints tracked in last frame into the current frame and search matches.
		// Used to track from previous frame (Tracking)
		int SearchByProjection(Frame& CurrentFrame, const Frame& LastFrame, const float th, const bool bMono);

		// Project MapPoints seen in KeyFrame into the Frame and search matches.
		// Used in relocalisation (Tracking)
		int SearchByProjection(Frame& CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*>& sAlreadyFound, const float th, const int ORBdist);

		// Project MapPoints using a Similarity Transformation and search matches.
		// Used in loop detection (Loop Closing)
		int SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float>& Scw, const std::vector<MapPoint*>& vpPoints, std::vector<MapPoint*>& vpMatched, int th, float ratioHamming = 1.0);

		// Project MapPoints using a Similarity Transformation and search matches.
		// Used in Place Recognition (Loop Closing and Merging)
		int SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float>& Scw, const std::vector<MapPoint*>& vpPoints, const std::vector<KeyFrame*>& vpPointsKFs, std::vector<MapPoint*>& vpMatched, std::vector<KeyFrame*>& vpMatchedKF, int th, float ratioHamming = 1.0);

		// Search matches between MapPoints in a KeyFrame and ORB in a Frame.
		// Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
		// Used in Relocalisation and Loop Detection
		int SearchByBoW(KeyFrame* pKF, Frame& F, std::vector<MapPoint*>& vpMapPointMatches);
		int SearchByBoW(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint*>& vpMatches12);

		// Matching for the Map Initialization (only used in the monocular case)
		int SearchForInitialization(Frame& F1, Frame& F2, std::vector<cv::Point2f>& vbPrevMatched, std::vector<int>& vnMatches12, int windowSize = 10);

		// Matching to triangulate new MapPoints. Check Epipolar Constraint.
		int SearchForTriangulation(KeyFrame* pKF1, KeyFrame* pKF2,
			std::vector<pair<size_t, size_t> >& vMatchedPairs, const bool bOnlyStereo, const bool bCoarse = false);

		// Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
		// In the stereo and RGB-D case, s12=1
		// int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);
		int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint*>& vpMatches12, const Sophus::Sim3f& S12, const float th);

		// Project MapPoints into KeyFrame and search for duplicated MapPoints.
		int Fuse(KeyFrame* pKF, const vector<MapPoint*>& vpMapPoints, const float th = 3.0, const bool bRight = false);

		// Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
		int Fuse(KeyFrame* pKF, Sophus::Sim3f& Scw, const std::vector<MapPoint*>& vpPoints, float th, vector<MapPoint*>& vpReplacePoint);

	public:

		static const int TH_LOW;
		static const int TH_HIGH;
		static const int HISTO_LENGTH;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	protected:
		float RadiusByViewingCos(const float& viewCos);

		void ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3);

		float mfNNratio;
		bool mbCheckOrientation;
	};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H