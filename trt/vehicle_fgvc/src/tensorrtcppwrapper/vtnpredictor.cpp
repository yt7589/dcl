#include "vtnpredictor.hpp"
#include "api_global.hpp"

#include <iostream>

const int IMG_W = 224;
const int IMG_H = 224;

void VTNPredictor::postProcess(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& net_outputs, std::vector<std::vector<float> >& out_results) const
{
	std::cout<<"net_outputs: picture_num="<<net_outputs.size()<<"; dim="<<net_outputs[0].size()<<";"<<std::endl;
	// 除以最小批次
	int batchsize = 8;
	int classNum = net_outputs[0].size()/batchsize;
	for (int in = 0; in < batchsize; ++in)
	{
		std::vector<float> rst;
		for (int j=0; j<classNum; j++)
		{
			rst.push_back(net_outputs[0][in*classNum + j]);
		}
		std::cout<<"rst.size:"<<rst.size()<<"; out_results.size:"<<out_results.size()<<"; s="<<out_results[0]<<";"<<std::endl;
		//double a = std::exp(net_outputs[0][2*in]);
		//double b = std::exp(net_outputs[0][1+2*in]);
		out_results.push_back(rst);
	}	
	



	for (auto &item:out_results)
	{
		item.pop_back();
	}
	//return;
		
//	for (auto item:out_results)
//	{
//		for (auto inner_item:item)
//		{
//		std::cout << inner_item << " " ;
//		}
//		std::cout << std::endl;
//	}



	
	
}

void VTNPredictor::postProcess(float* pgpu,int num,
        std::vector<std::vector<float> >& net_outputs,
        std::vector<std::vector<float> >& out_results) const
{
    std::vector<cv::Mat> images;
    postProcess(images, net_outputs, out_results);
}

void VTNPredictor::setConfig(const InputConfig& iconfig, WrapperConfig& config) const
{
	
	auto mean = 0.485*255;
	auto xFactor = 1/(0.225*255);
	//config.inputLayers.push_back("input_images:0");
	config.outputLayers.push_back("output");//596 , 698
	config.inputShape = std::vector<int>{-1,3,IMG_W,IMG_H};
	//config.modelInputType=0;  //0:  float; 1  uint8; 
	config.meanBGR=std::vector<float>{mean,mean,mean};
	config.xFactorBGR=std::vector<float>{xFactor, xFactor, xFactor};	
	
	BasePredictor::setConfig(iconfig, config);
}
	
