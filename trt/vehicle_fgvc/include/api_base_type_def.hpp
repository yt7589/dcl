#pragma once

#include <vector>
#include <string>

#include "api_base_type_def.hpp"

typedef struct __struct_input_config
{
	std::vector<int> devices;//"-1": cpu, "1" or "0,1" or "1,2,4" : gpu
	//for tensorrt
	std::string modelType; //"onnx", "caffe", "tehsorflow_pb" "mxnet" "engine" ...
	int maxBatchSize;
	int modelInputType; //0:  float; 1  fp16 ;  2  uint8; 
}InputConfig;


