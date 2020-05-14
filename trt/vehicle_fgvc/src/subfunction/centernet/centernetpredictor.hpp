#pragma once

#include "tensorrtpredictor.hpp"

class CenterNetPredictor: public TensorrtPredictor  //临时方案
{
public:
	CenterNetPredictor()=default;
	virtual ~CenterNetPredictor()=default;
private:
	virtual void postProcess(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& net_outputs, std::vector<std::vector<float> >& out_results) const override; //后处理
	virtual void setConfig(const InputConfig& iconfig, WrapperConfig& config) const override; //配置输入输出节点
	
	
	
};

class HelmetPredictor: public CenterNetPredictor  //临时方案
{
public:
	HelmetPredictor()=default;
	virtual ~HelmetPredictor()=default;
private:
	virtual std::vector<float> preProcess(const std::vector<cv::Mat> &images, const WrapperConfig& conf) const override;
	virtual void setConfig(const InputConfig& iconfig, WrapperConfig& config) const override final; //配置输入输出节点
	virtual void postProcess(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& net_outputs, std::vector<std::vector<float> >& out_results) const override; //后处理
};