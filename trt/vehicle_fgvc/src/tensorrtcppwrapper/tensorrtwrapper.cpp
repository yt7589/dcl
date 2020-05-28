#include "tensorrtwrapper.hpp"

#include <cassert>
#include <iostream>
#include <fstream>

#include <algorithm>



#include "Trt.hpp"
namespace {
}

// ******************************************  TensorRTwWarper  ******************************************
TensorRTWrapper::TensorRTWrapper()
{
}

TensorRTWrapper::~TensorRTWrapper()
{
}
void TensorRTWrapper::destroy_ipl()
{	
	_engine_data_save.clear();
	_pTinyTrt = nullptr;
}

std::vector<char> TensorRTWrapper::query_info(std::string key)
{ 
	if (key == "engine")
	{
		if (_engine_data_save.empty())
			std::cout << "data is empty. key = " << key << std::endl;
		return std::vector<char>(_engine_data_save.begin(), _engine_data_save.end());
	}
	else
	{
		std::cout << "warning, unknown key: " << key << std::endl;
		return std::vector<char>(); 
	}	
}

//std::string TensorRTWrapper::_engine_data_save = std::string("");
bool TensorRTWrapper::initial_ipl(const std::vector<std::vector<char>>& model_data,\
									const WrapperConfig& config) // 初始化
{
	if (!config.iConfig.devices.empty() )
	{
		//std::cout << "warning: make sure saved engine file match choosed device" << std::endl;
		//std::cout << "using device: " << config.iConfig.devices[0] << std::endl;
		if(config.iConfig.devices[0] >=0)
			_pTinyTrt->SetDevice(config.iConfig.devices[0]);
	}
	
	std::string model_str(model_data[0].begin(), model_data[0].end());

	
	_pTinyTrt = std::unique_ptr<Trt>(new Trt());
	if (config.iConfig.modelType == "onnx")
	{ 
		// since build engine is time consuming,so save we can serialize engine to file, it's much more faster
		if (_engine_data_save.empty())
			return _pTinyTrt->CreateEngine(model_str,
						 _engine_data_save, 
						 config.outputLayers, config.iConfig.maxBatchSize,\
                             config.iConfig.modelInputType); //698
		else 	return _pTinyTrt->CreateEngine("",
						 _engine_data_save, 
						 config.outputLayers, config.iConfig.maxBatchSize,\
                             config.iConfig.modelInputType); //698
	

	}
	else if (config.iConfig.modelType == "engine")
	{
		return _pTinyTrt->CreateEngine("",
						 model_str, 
						 config.outputLayers, config.iConfig.maxBatchSize,
                         config.iConfig.modelInputType); //698
	}
    else if (config.iConfig.modelType == "caffe")
    {
        assert(model_data.size() == 2);
        std::string model_str_caffe_model(model_data[1].begin(), model_data[1].end());
        std::function<std::vector<float>(int) > calib;
        
        if (_engine_data_save.empty())
        {
            return _pTinyTrt->CreateEngine(model_str, model_str_caffe_model,\
                             _engine_data_save, \
                             config.outputLayers,\
                             calib,\
                             config.iConfig.maxBatchSize,\
                             config.iConfig.modelInputType);
    
        }
         
         else
         {
             assert(0);
             return false;
         }
                             

    }
	else
	{
		//std::cout << "unkown config.modelType "<< config.iConfig.modelType << std::endl ;
		return false;
	}  
	
}

bool TensorRTWrapper::forward_ipl(const std::vector<float> &input, std::vector<std::vector<float> >& out_results) const
{
	_pTinyTrt->DataTransfer_inputAsync(input,0);

	// 0 for input index, you can get it from CreateEngine phase log output, True for copy input date to gpu
	_pTinyTrt->ForwardAsync();
	int outputIndex=1;
	out_results.resize(_config.outputLayers.size());
	//  get output.
    for (int i = 0 ; i < _config.outputLayers.size(); ++i)
	    _pTinyTrt->DataTransferAsync(out_results[i], outputIndex++, false); // you can get outputIndex in CreateEngine phase

    return true;
}
// old version
//bool TensorRTWrapper::forward_ipl(float* pGpuData, int len, std::vector<std::vector<float> >& out_results) const
//{
//	std::cout<<"TensorRTWrapper::forward_ipl 1"<<std::endl;
//	_pTinyTrt->DataTransfer_inputAsync(pGpuData, len,0);
//	std::cout<<"TensorRTWrapper::forward_ipl 2"<<std::endl;
//	// 0 for input index, you can get it from CreateEngine phase log output, True for copy input date to gpu
//	_pTinyTrt->ForwardAsync();
//	std::cout<<"TensorRTWrapper::forward_ipl 3"<<std::endl;
//	int outputIndex=1;
//	out_results.resize(_config.outputLayers.size());
//	std::cout<<"TensorRTWrapper::forward_ipl 4"<<std::endl;
//	//  get output.
//    for (int i = 0 ; i < _config.outputLayers.size(); ++i)
//	    _pTinyTrt->DataTransferAsync(out_results[i], outputIndex++, false); // you can get outputIndex in CreateEngine phase
//	std::cout<<"TensorRTWrapper::forward_ipl 5"<<std::endl;
//
//    return true;
//}


bool TensorRTWrapper::forward_ipl(float* pGpuData, int batch, std::vector<std::vector<float> >& out_results) const
{
    _pTinyTrt->inputPointerSet_Cao(pGpuData);
    _pTinyTrt->Forward_Cao(batch);
    int outputIndex = 1;
    out_results.resize(_config.outputLayers.size());
    for (int i = 0 ; i < _config.outputLayers.size(); ++i)
	    _pTinyTrt->DataTransferAsync(out_results[i], outputIndex++, false);
    _pTinyTrt->inputPointerRecover_Cao();
    return true;
}





