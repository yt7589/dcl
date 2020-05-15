
#include "predictor_api.hpp"
#include "vtnpredictor.hpp"

#include <iostream>
#include <cassert>

#include "ThreadPool.hpp"
#include <functional>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include "reflect.hpp"
//extern thread_local std::function<std::vector<float>(int)> g_calib;
bool PredictorAPI::init(std::vector<std::string> model_path, const InputConfig& iconfig) // 初始化
{	 
	std::cout<<"PredictorAPI::init0 1"<<std::endl;
     
	set(_pPredictor);
    //_pPredictor->init, _pPredictor, 
    auto a = _pPredictor.get();
	std::cout<<"PredictorAPI::init0 2"<<std::endl;
	if (iconfig.modelInputType == 2)
	{
		std::cout<<"PredictorAPI::init0 3"<<std::endl;
		assert(!_calibImages.empty());
		if (_calibImages.empty())
		{
			return false;
		}
		a->_calib = [this](int batchsize){return this->calib(batchsize);};
	}
	else
	{
		std::cout<<"PredictorAPI::init0 4"<<std::endl;
		assert(iconfig.modelInputType != 1);
		//iconfig.modelInputType = 0;
	}
    auto result = pool->enqueue([&model_path, &iconfig,a]\
            (){return a->init(model_path,iconfig ); });
          
	return result.get(); 
    //return _pPredictor->init(model_path, iconfig);
}
 
bool PredictorAPI::init(const std::vector<std::vector<char>>& model_data, const InputConfig& iconfig) // 初始化
{ 
	std::cout<<"PredictorAPI::init1 1"<<std::endl;
	set(_pPredictor);
	auto a = _pPredictor.get();
	std::cout<<"PredictorAPI::init 2"<<std::endl;
	if (iconfig.modelInputType == 2)
	{
		std::cout<<"PredictorAPI::init 3"<<std::endl;
		assert(!_calibImages.empty());
		if (_calibImages.empty())
		{
			return false;
		}
		a->_calib = [this](int batchsize){return this->calib(batchsize);};
	}
	else
	{
		std::cout<<"PredictorAPI::init 4"<<std::endl;
		assert(iconfig.modelInputType != 1);
		//iconfig.modelInputType = 0;
	}
	std::cout<<"PredictorAPI::init 5"<<std::endl;
	//g_calib = [this](int batchsize)(return this->calib(batchsize))
    //auto a = _pPredictor.get();
	//a->_calib = [this](int batchsize)(return this->calib(batchsize));
    auto result = pool->enqueue([&model_data, &iconfig,a]\
            (){return a->init(model_data,iconfig );});
	std::cout<<"PredictorAPI::init 6"<<std::endl;
             
	return result.get(); 
	//return _pPredictor->init(model_data, iconfig);
}


void PredictorAPI::setCalibImages(const std::vector<std::string> &images, int dstWidth, int dstHeight)
{
	_calibImages = images;
	_dstWidth = dstWidth;
	_dstHeight = dstHeight;
	if (dstWidth != 0)
	{
		assert(dstHeight != 0);
		
	}
}

std::vector<float> PredictorAPI::calib(int batchsize)
{
    std::cout <<_calibImages.size()<< " PredictorAPI::calib(int batchsize) " << batchsize << std::endl;
    std::vector<float> result;
    if (_calibImages.size() < batchsize)
    {
        return result;
    }

    int index = 0;
    std::vector<cv::Mat> imgs;
    while (index < batchsize  )
    {
        auto img_path = _calibImages.back();
        std::cout <<batchsize<<" == "<< index <<" read " << img_path << std::endl;
        auto img = cv::imread(img_path);
        if (_dstWidth != 0 && _dstHeight != 0)
        {
            if (_dstWidth != img.cols || _dstHeight != img.rows)
            {
                cv::resize(img, img, cv::Size(_dstWidth, _dstHeight), (0, 0), (0, 0));
            }
        }
        imgs.push_back(img);
        _calibImages.pop_back();
        index++;

    }
    std::vector<std::vector<float> >  out_results;
    _pPredictor->forward(imgs, out_results);
    //forward(imgs, out_results);
    assert(out_results.size()==1);
    return out_results[0];

}

PredictorAPI::PredictorAPI():pool(new ThreadPool(1))
{
}

PredictorAPI::PredictorAPI(std::string name):_register_name(name), pool(new ThreadPool(1))
{
    
}

PredictorAPI::~PredictorAPI()
{
    delete pool;
}

std::vector<char> PredictorAPI::query_info(std::string key)
{
    auto a = _pPredictor.get();
    auto result = pool->enqueue([&key,a]\
            (){return a->query_info(key);});
	return result.get(); 
	//return _pPredictor->query_info(key);
}

void PredictorAPI::forward(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& out_results)
{
    auto a = _pPredictor.get();
    auto result = pool->enqueue([&images,&out_results,a]\
            (){return a->forward(images,out_results );});
	result.get(); 
	//_pPredictor->forward(images, out_results);
}

void PredictorAPI::forward(float* pGpuData, int num,  std::vector<std::vector<float> >& out_results)
{
	std::cout<<"predictor_api.forward 1"<<std::endl;
    auto a = _pPredictor.get();
	std::cout<<"predictor_api.forward 2"<<std::endl;
    auto result = pool->enqueue([&pGpuData,&num,&out_results,a]\
            (){return a->forward(pGpuData,num,out_results );});
	std::cout<<"predictor_api.forward 3"<<std::endl;
	result.get(); 
	std::cout<<"predictor_api.forward 4"<<std::endl;
	//_pPredictor->forward(images, out_results);
}

void PredictorAPI::destroy()
{
	_pPredictor->destroy();
}

void PredictorAPI::set(std::unique_ptr<BasePredictor>& net)
{
    assert(false);
}


VTNPredictorAPI::VTNPredictorAPI()
{
}	

 VTNPredictorAPI::~VTNPredictorAPI()
{
}


void VTNPredictorAPI::set(std::unique_ptr<BasePredictor>& net)
{
	net = std::unique_ptr<BasePredictor>(new VTNPredictor());
}

//typedef VTNPredictorAPI task_head_tail;
//REGISTER(task_head_tail);
typedef VTNPredictorAPI task_vehicle_fgvc;
REGISTER(task_vehicle_fgvc);