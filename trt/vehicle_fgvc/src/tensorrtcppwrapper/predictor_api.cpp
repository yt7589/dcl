
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
	set(_pPredictor);
    //_pPredictor->init, _pPredictor, 
    auto a = _pPredictor.get();
	if (iconfig.modelInputType == 2)
	{
		assert(!_calibImages.empty());
		if (_calibImages.empty())
		{
			return false;
		}
		a->_calib = [this](int batchsize){return this->calib(batchsize);};
	}
	else
	{
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
	set(_pPredictor);
	auto a = _pPredictor.get();
	if (iconfig.modelInputType == 2)
	{
		assert(!_calibImages.empty());
		if (_calibImages.empty())
		{
			return false;
		}
		a->_calib = [this](int batchsize){return this->calib(batchsize);};
	}
	else
	{
		assert(iconfig.modelInputType != 1);
		//iconfig.modelInputType = 0;
	}
	//g_calib = [this](int batchsize)(return this->calib(batchsize))
    //auto a = _pPredictor.get();
	//a->_calib = [this](int batchsize)(return this->calib(batchsize));
    auto result = pool->enqueue([&model_data, &iconfig,a]\
            (){return a->init(model_data,iconfig );});
             
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
    auto a = _pPredictor.get();
    auto result = pool->enqueue([&pGpuData,&num,&out_results,a]\
            (){return a->forward(pGpuData,num,out_results );});
	result.get(); 
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