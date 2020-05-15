
#include <iostream>
#include <fstream>
#include "basepredictor.hpp"
#include "basewrapper.hpp"
#include <cassert>
#include "opencv2/opencv.hpp"
//#include "EntropyCalibrator.hpp" 

#include <functional>
thread_local std::function<std::vector<float>(int)>g_calib;

namespace
{
template<typename T>
bool readdata(const std::string &modelPath, std::vector<T>& bytes) //TODO 此函数应该使用基类中的实现
{
	assert(sizeof(T) == 1);
	std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
	
	if (!file.eof() && !file.fail()) {
		file.seekg(0, std::ios_base::end);
		std::streampos fileSize = file.tellg();
		bytes.resize(fileSize);
		//std::cout << "fileSize"<<fileSize<< " "<<std::endl; exit(0);
		
		file.seekg(0, std::ios_base::beg);
		file.read(static_cast<char*>(&bytes[0]), fileSize);
		return true;
	} else {
		return false;
	}
}

/*
1.

mat = Scalar(255,255,255) - mat;
2.

subtract(Scalar(255,255,255), mat, mat);
*/
}

std::vector<char> BasePredictor::query_info(std::string key)
{
	if (key == "batchsize")
	{
		return std::vector<char>{_config.iConfig.maxBatchSize};
	}
	return _pWrapper->query_info(key);
}
/*
std::vector<float> BasePredictor::calibFunctor(int batchsize)
{
	assert(!_calibImagesPath.empty());	
}
*/	
std::vector<float> BasePredictor::preProcess(const std::vector<cv::Mat> &images, const WrapperConfig& conf) const 
{
	
	float meanb = conf.meanBGR[0];
	float meang = conf.meanBGR[1];
	float meanr = conf.meanBGR[2];
	float xfactorb = conf.xFactorBGR[0];
	float xfactorg = conf.xFactorBGR[1];
	float xfactorr = conf.xFactorBGR[2];
	
	auto batch = images.size();//_config.inputShape[0];
	assert(batch <= conf.inputShape[0]);
	auto depth = conf.inputShape[1];
	auto height = conf.inputShape[2];
	auto width = conf.inputShape[3];
	// 批处理识别
	std::vector<cv::Mat> split_mats;
	
	

	std::vector<float> dataVec (batch*height*width*depth);
	float * dataPtr = dataVec.data();
	
	for (const auto &image : images)
	{
		//std::cout<<"image.rows: "<<image.rows<<"   height: "<<height<<std::endl;
		assert(image.rows == height);
		assert(image.cols == width);
		assert(image.type() == CV_8UC3 || image.type() == CV_32FC3 );
		//image.convertTo(tmp, CV_32FC3);
		//image = tmp-cv::Scalar(meanb,meang,meanr)
		
		std::vector<cv::Mat> channels;
		split(image,channels);
        //cv::imwrite("test.jpg", image);
        std::cout  << "prepross"<<std::endl;
        
		
        
		cv::Mat imageBlue(height, width, CV_32FC1, dataPtr);
		dataPtr += height * width ;
		cv::Mat imageGreen(height, width, CV_32FC1, dataPtr);
		dataPtr += height * width ;
		cv::Mat imageRed(height, width, CV_32FC1, dataPtr);
		dataPtr += height * width ;
		std::cout  << "1/xfactorb"<<1/xfactorb << " " << meanb <<std::endl;
        if(conf.swapBR)
        {
            channels.at(0).convertTo(imageRed, CV_32FC1, xfactorb, -meanb*xfactorb);
            channels.at(1).convertTo(imageGreen, CV_32FC1, xfactorg, -meang*xfactorg);
            channels.at(2).convertTo(imageBlue, CV_32FC1, xfactorr, -meanr*xfactorr);
        } 
        else
        {
            channels.at(0).convertTo(imageBlue, CV_32FC1, xfactorb, -meanb*xfactorb);
            channels.at(1).convertTo(imageGreen, CV_32FC1, xfactorg, -meang*xfactorg);
            channels.at(2).convertTo(imageRed, CV_32FC1, xfactorr, -meanr*xfactorr);
        }
        
		
	}
    std::cout  << "prepross"<<dataVec[100*height] << " " <<dataVec[100*width+1] << "meanb "<< meanb <<std::endl;

    
	return dataVec;
}


BasePredictor::BasePredictor()
{
}


BasePredictor::~BasePredictor()
{
	std::cout << " xx ";
	destroy();
	std::cout << " sss ";
}	
	
void BasePredictor::destroy()
{
	//_pWrapper->destroy();
	std::cout << " == "<<std::endl;
	_pWrapper = nullptr;
}
	

	
void BasePredictor::forward(float* pGpuData, int num, std::vector<std::vector<float> >& out_results) const
{
	std::cout<<"BasePredictor.forward 1"<<std::endl;
	if (num == 0)
		return;
	std::cout<<"BasePredictor.forward 2"<<std::endl;
	
	if (num <= _config.iConfig.maxBatchSize)
	{
		std::cout<<"BasePredictor.forward 3"<<std::endl;
		forward_full(pGpuData, num, out_results);
		return ;
	}
	else
	{
		std::cout<<"BasePredictor.forward 4"<<std::endl;
		std::cout << "error ,num <= _config.iConfig.maxBatchSize "<< 
				num <<" vs. "<<_config.iConfig.maxBatchSize<< std::endl;
		assert(false);
		
		return;
	}
	
	// std::vector<cv::Mat> one_round;
	// for (auto iter = images.begin(); iter != images.end(); ++iter)
	// {
	// 	one_round.push_back(*iter);
	// 	if (one_round.size() == _config.iConfig.maxBatchSize || iter == images.end()-1)
	// 	{
	// 		int pre_size = out_results.size();
	// 		forward_full(one_round, out_results);
	// 		//std::cout <<pre_size << " "<< one_round.size() << " "<< out_results[0].size()<<std::endl;
	// 		assert(pre_size + one_round.size() == out_results.size());
	// 		one_round.clear();
	// 	}
	// }
}

void BasePredictor::forward(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& out_results) const
{
	if (images.empty())
		return;
	std::cout << "forward images.size()  "<<images.size() <<std::endl;
	
	if (images.size() <= _config.iConfig.maxBatchSize)
	{
		forward_full(images, out_results);
		return ;
	}
	std::vector<cv::Mat> one_round;
	for (auto iter = images.begin(); iter != images.end(); ++iter)
	{
		one_round.push_back(*iter);
		if (one_round.size() == _config.iConfig.maxBatchSize || iter == images.end()-1)
		{
			int pre_size = out_results.size();
			forward_full(one_round, out_results);
			//std::cout <<pre_size << " "<< one_round.size() << " "<< out_results[0].size()<<std::endl;
			assert(pre_size + one_round.size() == out_results.size());
			one_round.clear();
		}
	}
}


void BasePredictor::forward_full(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& out_results) const
{
    if (images.size() > _config.iConfig.maxBatchSize)
    {
        std::cout << "error: (images.size() > _config.iConfig.maxBatchSize)  exit now" << std::endl;
        return;
    }
	auto input = preProcess(images, _config);
    std::cout << "_config.iConfig.modelInputType  "<<_config.iConfig.modelInputType <<std::endl;
	if (_config.iConfig.modelInputType == 2 && _config.iConfig.modelType != "engine")
	{
		
		out_results.push_back(input);
		return;

	}
	std::vector<std::vector<float> > tmp_out_results;
	_pWrapper->forward(input, tmp_out_results);
	postProcess(images, tmp_out_results, out_results);
}


void BasePredictor::forward_full(float* pGpuData, int num, std::vector<std::vector<float> >& out_results) const
{
	std::cout<<"BasePredictor.forward_full 1"<<std::endl;
    if (num > _config.iConfig.maxBatchSize)
    {
		std::cout<<"BasePredictor.forward_full 2"<<std::endl;
        std::cout << "error: (images.size() > _config.iConfig.maxBatchSize)  exit now" << std::endl;
        return;
    }
	//auto input = preProcess(images, _config);

	// if (_config.iConfig.modelInputType == 2 && _config.iConfig.modelType != "engine")
	// {
		
	// 	out_results.push_back(input);
	// 	return;

	// }
	std::cout<<"BasePredictor.forward_full 3"<<std::endl;
	std::vector<std::vector<float> > tmp_out_results;
	std::cout<<"BasePredictor.forward_full 4"<<std::endl;
	_pWrapper->forward(pGpuData, num, tmp_out_results);
	std::cout<<"BasePredictor.forward_full 5"<<std::endl;
	postProcess(pGpuData, num, tmp_out_results, out_results);
}



/*
virtual  std::vector<float> BasePredictor::preProcess(const std::vector<cv::Mat> &images, const WrapperConfig& conf)
{
	return convertMats(images, conf.meanBGR, conf.xFactorBGR);
}
*/

bool BasePredictor::init(std::vector<std::string> model_path, const InputConfig& iconfig)
{
	
//	g_calib = _calib;
	assert (!model_path.empty());
	std::vector<std::vector<char>> bytes(model_path.size());
    for (int i = 0; i < model_path.size(); ++i)
    {
        if (model_path[i].size() < 2048)
        {
            //std::cout << model_path[i];exit(0);
            auto status_load = readdata(model_path[i], bytes[i]);
			//std::cout << bytes[i].size();exit(0);
            if (!status_load)
            {
                std::cout << "ERROR: Loading model failed... " << model_path[i] << std::endl;
                return false;
            }
        } else{
            bytes[i].resize(model_path[i].size());
            bytes[i].assign(model_path[i].begin(), model_path[i].end());
        }

    }
	
	//std::vector<char>tmp (model_path.begin(), model_path.end());
	//bytes[0]=tmp;
	return init(bytes, iconfig);
}

bool BasePredictor::init(const std::vector<std::vector<char>>& model_data, const InputConfig& iconfig)
{	 
	//_pWrapper = std::make_unique<TensorRTWrapper>();
	setWrapper(_pWrapper);
	g_calib = _calib;	
	
	if (!_pWrapper)
	{
		std::cout << "error : setWrapper failed.. "<<std::endl; 
		return false;
	}
	setConfig(iconfig, _config);
	return _pWrapper->initial(model_data, _config);	
}
	
	

void BasePredictor::setConfig(const InputConfig& iconfig, WrapperConfig&conf) const 
{
	conf.iConfig = iconfig;
	if (conf.meanBGR.empty())
	{
		conf.meanBGR.push_back(0.0);
	}
	
	if (conf.xFactorBGR.empty())
	{
		conf.xFactorBGR.push_back(1.0);
	}
	
	while (conf.meanBGR.size() < 3)
		conf.meanBGR.push_back(conf.meanBGR[0]);
	while (conf.xFactorBGR.size() < 3)
		conf.xFactorBGR.push_back(conf.xFactorBGR[0]);


}
