#pragma once

#include <vector>
#include <string>

#include "api_base_type_def.hpp"

class WrapperConfig
{
    public:
    WrapperConfig():swapBR(false){}
	InputConfig iConfig;	
	
	//for tensorrt
	std::vector<std::string> inputLayers;//TODO : now inputLayers.size() must be one
	std::vector<std::string> outputLayers;
	
	std::vector<int> inputShape; //[batch], [depth], [height], [width],   ([] means optinal, for example {1,224,224,3}, {3}, {224,224,3})
									
	//int modelInputType; //0:  float; 2  uint8; 1  fp16
	
	std::vector<float> meanBGR;
	std::vector<float>  xFactorBGR;
    bool swapBR;
	

};




