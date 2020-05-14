#pragma once

#include "tensorrtpredictor.hpp"

class VTNPredictor: public TensorrtPredictor  //临时方案
{
public:
	VTNPredictor()=default; 
	virtual ~VTNPredictor()=default;
private:
	virtual void postProcess(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& net_outputs, std::vector<std::vector<float> >& out_results) const override final; //后处理
    virtual void postProcess(float*pgpu,int num, std::vector<std::vector<float> >& net_outputs, std::vector<std::vector<float> >& out_results) const override final; //后处理


	virtual void setConfig(const InputConfig& iconfig, WrapperConfig& config) const override final; //配置输入输出节点
	
	
	
};