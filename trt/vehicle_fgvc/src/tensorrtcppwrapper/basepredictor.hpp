#pragma once

#include <vector>
#include <memory>
#include "opencv2/core/core.hpp"
class BaseWrapper;
#include <functional>
#include "wrapper_base_type_def.hpp"
class BasePredictor  //临时方案
{
public:
	BasePredictor();
	virtual ~BasePredictor();
	
	bool init(std::vector<std::string> model_path, const InputConfig& iconfig); // 初始化
	bool init(const std::vector<std::vector<char>>& model_data, const InputConfig& iconfig); // 初始化
	
	void forward(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& out_results) const;
	void forward(float* pGpuData, int num, std::vector<std::vector<float> >& out_results) const;
	void destroy();
	
	std::vector<char> query_info(std::string key);
	//std::vector<std::string> _calibImagesPath;
	std::function<std::vector<float>(int)> _calib;
protected:
	virtual std::vector<float> preProcess(const std::vector<cv::Mat> &images, const WrapperConfig& conf) const ;
	//std::vector<float> calibFunctor(int batchsize);

	virtual void setConfig(const InputConfig& iconfig, WrapperConfig& config) const = 0; //配置输入输出节点	
private:
	/*子类需实现的功能*/
	virtual void postProcess(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& net_outputs,\
							std::vector<std::vector<float> >& out_results) const {} //后处理
	virtual void postProcess(float* pGpuData, int num, std::vector<std::vector<float> >& net_outputs,\
							std::vector<std::vector<float> >& out_results) const {} //后处理						
	
	virtual void setWrapper(std::unique_ptr<BaseWrapper>& pWrapper) = 0;
	void forward_full(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& out_results) const;
	void forward_full(float* pGpuData, int num, std::vector<std::vector<float> >& out_results) const;

	WrapperConfig _config;
	std::unique_ptr<BaseWrapper> _pWrapper;
};
