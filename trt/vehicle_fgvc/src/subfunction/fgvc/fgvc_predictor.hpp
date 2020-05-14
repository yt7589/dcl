#pragma once

#include "tensorrtpredictor.hpp"

class FgvcPredictor: public TensorrtPredictor  //临时方案
{
public:
	FgvcPredictor()=default;
	virtual ~FgvcPredictor()=default;
private:
	virtual void postProcess(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& net_outputs, std::vector<std::vector<float> >& out_results) const override; //后处理
	virtual void setConfig(const InputConfig& iconfig, WrapperConfig& config) const override; //配置输入输出节点
	
	
	
};

class FgvcVehiclePredictor: public FgvcPredictor  //临时方案
{
public:
	FgvcVehiclePredictor()=default;
	virtual ~FgvcVehiclePredictor()=default;
private:
	virtual std::vector<float> preProcess(const std::vector<cv::Mat> &images, const WrapperConfig& conf) const override;
	virtual void setConfig(const InputConfig& iconfig, WrapperConfig& config) const override final; //配置输入输出节点
	virtual void postProcess(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& net_outputs, std::vector<std::vector<float> >& out_results) const override; //后处理
};