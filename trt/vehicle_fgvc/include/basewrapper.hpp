#pragma once
/*
warper for headers of tensorflow, tensorrt, 
tensorflow的头文件太多太复杂，不需要暴露在外。用此中间层库包裹这些头文件。
*/
#include "wrapper_base_type_def.hpp"
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>




class BaseWrapper
{
public:
	BaseWrapper();
	virtual ~BaseWrapper();
	
	/************对外公共接口**********************************************************************************************/
	/*初始化，重复初始化时会先自动destroy前一次的初始化*/
	bool initial(const std::vector<std::vector<char>>& model_data,const WrapperConfig& config); // 初始化
	
	/* 手动释放。 在initial或 析构时会自动调用此函数*/
	void destroy();
	
	/* 前向*/
	bool forward(const std::vector<float> &input,\
						 std::vector<std::vector<float> >& out_results);
	bool forward(float* pGpuData, int num,\
						 std::vector<std::vector<float> >& out_results);					 
						 
	//bool initial(std::string model_path, const WrapperConfig& config); // 初始化
	//virtual bool forward(const std::vector<char> &input, std::vector<std::vector<float> >& out_results) const;
	virtual std::vector<char> query_info(std::string key){return std::vector<char>();}					 
protected:
	/*共享成员变量*/
	WrapperConfig _config; 
		
	
private:
	/*子类需实现的功能*/
	virtual bool initial_ipl(const std::vector<std::vector<char>>& model_data,const WrapperConfig& config)=0; // 初始化
	virtual void destroy_ipl()=0;
	virtual bool forward_ipl(const std::vector<float> &images,\
						 std::vector<std::vector<float> >& out_results) const
						  {assert(false);return false;};
	virtual bool forward_ipl(float* pGpuData, int num,\
						 std::vector<std::vector<float> >& out_results) const
						  {assert(false);return false;};					 
	
	
	/*私有状态变量*/
	bool _bInitialized;
};
