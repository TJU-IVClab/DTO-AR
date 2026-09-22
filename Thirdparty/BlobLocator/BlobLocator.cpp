#include <opencv2/opencv.hpp>
#include "BlobLocator.h"


cv::Rect expandBoundingBox(const cv::Rect& boundingBox, const cv::Size& imageSize, int padding) {
    int x = std::max(0, boundingBox.x - padding);
    int y = std::max(0, boundingBox.y - padding);
    int width = std::min(imageSize.width - x, boundingBox.width + 2 * padding);
    int height = std::min(imageSize.height - y, boundingBox.height + 2 * padding);
    return cv::Rect(x, y, width, height);
}


void main() {


	cv::Mat image = cv::imread("photoneo_raw//7_IMG_Texture_R.tif", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH); // 以灰度模式读取

    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return;
    }

    cv::Mat src;
    image.convertTo(src, CV_8U, 1.0);

    cv::Mat m_canny_mat;


    int m_iAperture = 5;
    double m_CannyLowThreshold = 30 * 16;
    double m_CannyHighThreshold = 90 * 16;


    cv::Canny(src, m_canny_mat, m_CannyLowThreshold, m_CannyHighThreshold, m_iAperture, true);


    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(m_canny_mat, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);


    cv::Mat contourImage = cv::Mat::zeros(src.size(), CV_8UC3);

    std::vector <cv::Point2f> get_location_result;
    get_location_result.reserve(contours.size());

    float ratio_tol = 3.0;

    for (size_t i = 0; i < contours.size(); i++)
    {
        if (hierarchy[i][2] != -1)
        {
            continue;
        }

        cv::Rect boundingBox = cv::boundingRect(contours[i]);

        float ratio = (float)boundingBox.width / (float)boundingBox.height;

        if (ratio > ratio_tol || ratio_tol < 1 / ratio_tol)
        {
            continue;
        }

        int area = boundingBox.area();
        if (area > 500 || area < 20)
        {
            continue;
        }

        boundingBox = expandBoundingBox(boundingBox, src.size(), 3);

        // 创建ROI掩膜
        cv::Mat mask = cv::Mat::zeros(boundingBox.size(), CV_8UC1);
        cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, -boundingBox.tl());
        cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), 4, cv::LINE_8, cv::noArray(), INT_MAX, -boundingBox.tl());

        // 提取ROI中的灰度图像
        cv::Mat grayROI = src(boundingBox);

        // 应用掩膜，确保mask与grayROI尺寸一致
        cv::Mat maskedGrayROI;
        cv::bitwise_and(grayROI, grayROI, maskedGrayROI, mask);

        cv::Mat _src = maskedGrayROI;

        int roiWidth = boundingBox.width;
        int roiHeight = boundingBox.height;

        POB* iPob = new POB;
        IMWIN* iImwin = new IMWIN;
        LOCATE* iLocate = new LOCATE;

        iImwin->h_win = roiWidth;
        iImwin->v_win = roiHeight;
        iImwin->cur_x = iImwin->h_win / 2;
        iImwin->cur_y = iImwin->v_win / 2;

        iImwin->ucBuf = new unsigned char* [_src.rows];
        for (int i = 0; i < _src.rows; ++i) {
            iImwin->ucBuf[i] = _src.ptr<unsigned char>(i);
        }


        //iLocate->locator = '3';
        iLocate->locator = '2';
        iLocate->shape = 'c';
        iLocate->threshold = 'p';
        iLocate->range = 20;

        iLocate->bw_test = 0;
        iLocate->ratio_test = 1;

        iLocate->ratio = 3.0;
        iLocate->min_targ_size = 3;

        iLocate->level = 2.0;


        int rThreshold = -1;
        long rXsize = -1;
        long rYsize = -1;

        short Rnt = get_location(iPob, iImwin, iLocate, &rThreshold, &rXsize, &rYsize);

        if (Rnt == 0)
        {

            get_location_result.push_back(cv::Point2f(iPob->pob_pix[0], iPob->pob_pix[1]) + cv::Point2f(boundingBox.tl().x, boundingBox.tl().y));

            cv::Scalar randomColor(rand() % 256, rand() % 256, rand() % 256);
            cv::drawContours(contourImage, contours, static_cast<int>(i), randomColor, 1, 8, hierarchy);
        }
        else
        {
            std::cout << Rnt << std::endl;
        }

    }


    return;





}