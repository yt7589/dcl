
#pragma once
#include <vector>
#include "opencv2/core/core.hpp"

namespace ImgUtils
{
cv::Mat keepRatioUpsample(const cv::Mat &img,\
									int max_h, int max_w,\
									bool swapRB = false);


std::vector<cv::Mat> resizeAndPadding(const std::vector<cv::Mat> &images,
										int input_h,int input_w, 
										std::string padding_location = "center", //center||topleft ||bottomrigh
										bool swapRB = false);
}