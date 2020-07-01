#include "basewrapper.hpp"

#include <cassert>
#include <iostream>
//#include <fstream>

#include <algorithm>
// ******************************************  BaseWrapper  ******************************************

BaseWrapper::BaseWrapper():_bInitialized(false)
{
}

BaseWrapper::~BaseWrapper()
{
}

void BaseWrapper::destroy()
{  //std::cout << " hh ";
	destroy_ipl();
	_bInitialized = false;
}

bool BaseWrapper::initial(const std::vector<std::vector<char>>& model_data,const WrapperConfig& config)
{
	std::cout<<"BaseWrapper::initial Ln26 1"<<std::endl;
	if (_bInitialized)
	{
		std::cout << "warning: BaseWrapper has areadly been initialized, starting reinitializing..."  << std::endl;
		destroy();
		//return _bInitialized;
	}
	std::cout<<"BaseWrapper::initial 2"<<std::endl;
	// initial base class
	_config = config;
	if (_config.inputLayers.size() > 1)
	{
		std::cout << "ERROR: inputLayers.size() must be smaller than 1..."  << std::endl;
		return false;
	}
	std::cout<<"BaseWrapper::initial 3"<<std::endl;
	std::vector<int> devices = config.iConfig.devices;
	if (devices.empty())
	{
		devices.push_back(-1);
	}
	std::cout<<"BaseWrapper::initial 4"<<std::endl;
	auto maxPosition = std::max_element(devices.begin(), devices.end());
	if (*maxPosition > 16)
	{
		std::cout<< "such a big deice : " << *maxPosition <<std::endl;
		return false;
	}
	std::cout<<"BaseWrapper::initial 5"<<std::endl;
	_config.iConfig.devices = devices;
	_bInitialized = initial_ipl(model_data, config);
	std::cout<<"BaseWrapper::initial 6"<<std::endl;
	return _bInitialized;
}

bool BaseWrapper::forward(const std::vector<float> &input,\
						 std::vector<std::vector<float> >& out_results)
{
	if (!_bInitialized)
	{
		std::cout << "Error: BaseWrapper has not been initialized..."  << std::endl;
		return false;
	}
	forward_ipl(input, out_results);
}

// old version
//bool BaseWrapper::forward(float* pGpuData, int num,\
//						 std::vector<std::vector<float> >& out_results)
//{
//	std::cout<<"BaseWrapper::forward 1"<<std::endl;
//	int len = num * _config.inputShape[1] * _config.inputShape[2]*
//			 _config.inputShape[3];
//	std::cout<<"BaseWrapper::forward 2"<<std::endl;
//	if (!_bInitialized)
//	{
//		std::cout<<"BaseWrapper::forward 3"<<std::endl;
//		std::cout << "Error: BaseWrapper has not been initialized..."  << std::endl;
//		return false;
//	}
//	std::cout<<"BaseWrapper::forward 4"<<std::endl;
//	forward_ipl(pGpuData, len, out_results);
//	std::cout<<"BaseWrapper::forward 5"<<std::endl;
//}

bool BaseWrapper::forward(float* pGpuData, int num,\
						 std::vector<std::vector<float> >& out_results)
{

    if (!_bInitialized)
    {
        std::cout<<"BaseWrapper::forward 3"<<std::endl;
        std::cout << "Error: BaseWrapper has not been initialized..."  << std::endl;
        return false;
    }
    forward_ipl(pGpuData, num, out_results);
}