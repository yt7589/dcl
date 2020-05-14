#pragma once


#include "basepredictor.hpp"

class TensorrtPredictor: public BasePredictor  //临时方案
{
public:
	TensorrtPredictor(){};
	virtual ~TensorrtPredictor(){};
private:
	//virtual void postProcess(std::vector<std::vector<float> >& net_outputs, std::vector<std::vector<float> >& out_results) const = 0; //后处理
	//virtual void setConfig(std::vector<int> devices, WrapperConfig& config) const = 0; //配置输入输出节点
	
	
	virtual void setWrapper(std::unique_ptr<BaseWrapper>& pWrapper) override final;
	
};