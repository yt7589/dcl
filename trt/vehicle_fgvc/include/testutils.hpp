//
// Created by cao on 19-10-26.
//
#pragma once
#include "testutils.hpp"
#include <string>
#include <iostream>
#include <memory>


#include "opencv2/opencv.hpp"

#include "predictor_api.hpp"


void SaveEngine(const std::string& fileName, std::vector<char> &data);
#include <sys/time.h>

double getTime();
std::vector<cv::Mat> getinputimg();

std::vector<std::string> loadList(const std::string& listfile);

void getFileList(std::string dirPath, std::vector<std::string>& pathList);



class RaiiTimer
{
	public:
	void start()
	{
		timer= getTime();
	}
	
	RaiiTimer()
	{
		start();
	}
	
	void stop(std::string message = "")
	{
		std::cout <<" message: " << message << " " << "Time used: "<< getTime()- timer <<  std::endl;
	}
	
	void restart(std::string message = "")
	{
		std::cout <<" message: " << message << " " << "Time used: "<< getTime()- timer <<  std::endl;
		start();
	}
	
	double timer;
	
	
};