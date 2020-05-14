//
// Created by cao on 19-10-26.
//


#include <string>
#include <iostream>
#include <memory>


#include "opencv2/opencv.hpp"

#include "predictor_api.hpp"
#include "testutils.hpp"

int testvtn(int argc, const char** argv){

	VTNPredictorAPI eng;
	InputConfig iconfig;
	iconfig.devices.push_back(0);
	iconfig.modelType = "onnx";
	iconfig.maxBatchSize = 16;
	// create engine and running context, note that engine file is device specific, so don't copy engine file to new device, it may cause crash
		

     //eng.init("./model/resnet34_vtn." +iconfig.modelType,
     std::vector<std::string> models {"./model/resnet34_vtn." +iconfig.modelType};
     eng.init(models,
     
    //eng.init("/data2/zhangsy/head/c3ae/model/head_recog." +iconfig.modelType,

					 iconfig // since build engine is time consuming,so save we can serialize engine to file, it's much more faster
					);//698
					//std::cout << "error : s failed.. "<<std::endl; 
	// trt.CreateEngine(onnxModel,engineFile,maxBatchSize); // for onnx model
	// trt.CreateEngine(uffModel, engineFile, inputTensorName, inputDims, outputTensorName, maxBatchSize); // for tensorflow model
	
	
	auto inputs = getinputimg();
	std::vector<std::vector<float> > out_results;
	for (int i = 0; i < 1; ++i)
		eng.forward(inputs, out_results);
	
	auto en = eng.query_info("engine");
	if(!en.empty());
		SaveEngine("./model/resnet34_vtn.engine", en);
	
	for (int i = 0; i < 10; ++i)
	{
		out_results.clear();
		auto start =  getTime();
		
		eng.forward(inputs, out_results);
	std::cout << " TIME " << getTime()- start<< std::endl;
	}
	// you might need to do some pre-processing in input such as normalization, it depends on your model.
	std::cout << " input.size() " << inputs[0].size() << std::endl;
	std::cout << " out_results " << out_results.size() << en.size() << std::endl;
    return 0;
}
