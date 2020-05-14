#pragma once
/*
warper for tensorrt's header
tensorrt的头文件太多太复杂，不需要暴露在外。用此中间层库包裹这些头文件。之后需要功能正交的继承Predictor类，
*/
#include "basewrapper.hpp"

#include <memory>

class Trt;

class TensorRTWrapper: public BaseWrapper
{
public:
	TensorRTWrapper();
	std::string _engine_data_save;
	
	//bool initial(std::string model_path, const WrapperConfig& config); // 初始化
private:	
	std::vector<char> query_info(std::string key) override;
	
	bool initial_ipl(const std::vector<std::vector<char>>& model_data,const WrapperConfig& config) override; // 初始化
	void destroy_ipl() override;

	bool forward_ipl(const std::vector<float> &input, std::vector<std::vector<float> >& out_results) const override;
	bool forward_ipl(float* pGpuData, int len, std::vector<std::vector<float> >& out_results) const override;

	~TensorRTWrapper();
	
	std::unique_ptr<Trt>  _pTinyTrt;
	
};
