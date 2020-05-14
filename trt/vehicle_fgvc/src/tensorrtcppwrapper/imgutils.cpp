#include "imgutils.hpp"
#include "opencv2/imgproc/imgproc.hpp"


namespace ImgUtils
{
cv::Mat keepRatioUpsample(const cv::Mat &img,
									int max_h, int max_w,
									bool swapRB)
{
	float scale = cv::min(float(max_w)/img.cols,float(max_h)/img.rows);
	auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);
	
	cv::Mat resized;
	
	if (swapRB)
	{
		if (scale >= 1.0)
		{
			cv::cvtColor(img, resized, cv::COLOR_RGB2BGR);
			cv::resize(resized, resized, scaleSize,0,0);
		}
		else
		{
			cv::resize(img, resized, scaleSize,0,0);
			cv::cvtColor(resized, resized, cv::COLOR_RGB2BGR);
		}
	}
	else
	{
		cv::resize(img, resized, scaleSize,0,0);
	}
	
	return resized;
}
									
std::vector<cv::Mat> resizeAndPadding(const std::vector<cv::Mat> &images,  
									int input_h,int input_w, 
									std::string padding_location,//center||topleft ||bottomrigh
									bool swapRB) 
{	
	std::vector<cv::Mat> results;

	for (const auto &img : images)
	{

		float scale = cv::min(float(input_w)/img.cols, float(input_h)/img.rows);
		auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);
		
		cv::Mat resized = keepRatioUpsample(img, input_h, input_w, swapRB);

		cv::Mat cropped = cv::Mat::zeros(input_h,input_w,CV_8UC3);
		auto startx = 0;
		auto starty = 0;
		if (padding_location == "center")
		{
			startx = (input_w- scaleSize.width)/2;
		    starty = (input_h-scaleSize.height)/2;
		}
		else if (padding_location == "topleft")
		{
		}
		else if (padding_location == "bottomrigh")
		{
			startx = (input_w- scaleSize.width);
		    starty = (input_h-scaleSize.height);
		}
		else
		{
			assert(false);
		}
		
		cv::Rect rect(startx, starty, scaleSize.width,scaleSize.height);
		resized.copyTo(cropped(rect));
		results.push_back(cropped);
	}
	
	return results;
}


}