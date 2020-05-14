#pragma once

#include "api_base_type_def.hpp"
#include "opencv2/core/core.hpp"

class BasePredictor;
class ThreadPool;
#include <memory>

#include<functional>
class PredictorAPI  //临时方案
{
public:
	PredictorAPI();	
    PredictorAPI(std::string name);		
	virtual ~PredictorAPI();	
	bool init(std::vector<std::string> model_path, const InputConfig& iconfig); // 初始化
	bool init(const std::vector<std::vector<char>>& model_data, const InputConfig& iconfig); // 初始化
	
	void forward(const std::vector<cv::Mat> &images, std::vector<std::vector<float> >& out_results);
    void forward(float* pGpuData, int num, std::vector<std::vector<float> >& out_results);
	void destroy();
	std::vector<char> query_info(std::string key);
	virtual void setCalibImages(const std::vector<std::string> &images, int dstWidth = 0, int dstHeight = 0);


    std::string _register_name;


private:	
	int _dstWidth;
	int _dstHeight;
	std::vector<float> calib(int batchsize);

	virtual void set(std::unique_ptr<BasePredictor>& net);	
	std::unique_ptr<BasePredictor> _pPredictor;
    ThreadPool *pool;
	std::vector<std::string> _calibImages;
    
};

class VTNPredictorAPI :public PredictorAPI//临时方案
{
public:
	VTNPredictorAPI();		
	virtual ~VTNPredictorAPI();
private:	
	virtual void set(std::unique_ptr<BasePredictor>& net) override;
};

class CenterPosePredictorAPI :public PredictorAPI//临时方案
{
public:
	struct Box{
    float x1;
    float y1;
    float x2;
    float y2;
};
struct landmarks{
    float x;
    float y;
};

struct HumanPose{ 
    //x1 y1 x2 y2
    Box bbox;
    //float objectness;
    int classId;
    float prob;
    landmarks marks[17];
};

struct HumanPoseResult{
    //x1 y1 x2 y2
    Box bbox;
    //float objectness;
    float prob;
    landmarks marks[17];
	int attr[17]; //0 : 可见 1 other
};



struct heatmaps{
    float prob;
	landmarks mark;
};
private:	

	virtual void set(std::unique_ptr<BasePredictor>& net) override;
};
