//
// Created by cao on 19-10-26.
//


#include <string>
#include <iostream>
#include <memory>


#include "opencv2/opencv.hpp"

#include "centernet_predictor_api.hpp"

#include "testutils.hpp"


using Det = CenterNetPredictorAPI::Det;


int testcenternet(int argc, const char** argv){
	CenterNetPredictorAPI eng;

    InputConfig iconfig;
	//iconfig.devices.push_back(0); 
	iconfig.modelType = "onnx";
	iconfig.maxBatchSize = 1;
    iconfig.modelInputType = 0;

    if (iconfig.modelInputType == 2)
    {
        std::string calib = "./calib/calib_phonecall.txt";

        
        std::vector<std::string> calib_imgs = loadList(calib);
        eng.setCalibImages(calib_imgs, 224, 224);

        std::cout << " calib imgs: " << calib_imgs.size() << std::endl;
    }
	
	// create engine and running context, note that engine file is device specific, so don't copy engine file to new device, it may cause crash
		
    std::cout << "init start" << std::endl;
	std::string	modelPath = "/home/zhouxd/Code/alg-mx_temp/model/n,3,224,224_smoke.";
    modelPath = "./model/n,3,224,224_phone.";
    if (iconfig.modelInputType == 2)
    {
        modelPath = "./1,3,224,224_phone.";
    }
    
    std::vector<std::string> models {modelPath +iconfig.modelType}; 
     eng.init(models,
    //eng.init("/data2/zhangsy/head/c3ae/model/head_recog." +iconfig.modelType,

					 iconfig // since build engine is time consuming,so save we can serialize engine to file, it's much more faster
					);//698
					//std::cout << "error : s failed.. "<<std::endl; 
	// trt.CreateEngine(onnxModel,engineFile,maxBatchSize); // for onnx model
	// trt.CreateEngine(uffModel, engineFile, inputTensorName, inputDims, outputTensorName, maxBatchSize); // for tensorflow model
	std::cout << "init finished" << std::endl;
	std::string pathstr = "test_pic/13_2.jpg";
 
	auto img = cv::imread(pathstr);
    cv::resize(img,img,cv::Size(224,224));
	std::vector<cv::Mat> inputs; 
    for (int i = 0; i < iconfig.maxBatchSize; ++i)
	    inputs.push_back(img);
	std::vector<std::vector<float> > out_results;
    std::cout << " cc " << std::endl; 
	for (int i = 0; i < 1; ++i)
		eng.forward(inputs, out_results);
	std::cout << " bb " << std::endl; 

    int mark;
    int box_think = (img.rows + img.cols) * .001*0.0001;
    float label_scale = img.rows * 0.0009;
    int base_line;
    std::cout << out_results.size() << std::endl;
    const auto &item = out_results[0];
    if (item.empty())
    {
        std::cout << " not detected "<<std::endl;
        return 0;
    }
    {
        const Det* pDet = reinterpret_cast<const Det*>((char*)item.data());
        std::vector<cv::Scalar> color = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0)};
        std::string label;
        std::stringstream stream;
        stream << std::to_string(1) << " " << pDet->prob << std::endl;
        std::getline(stream, label);

        auto size = cv::getTextSize(label, cv::FONT_HERSHEY_COMPLEX, label_scale, 1, &base_line);

        cv::rectangle(img, cv::Point(pDet->x1, pDet->y1),
                      cv::Point(pDet->x2, pDet->y2), 
                      color[0], box_think * 2, 8, 0);
                      auto det = *pDet;
                       std::cout << " det.x1 " << det.x1 << " "
             <<det.y1<<" "  << det.x2 << " " <<det.y2  << std::endl;
        {
            cv::putText(img, label,
                        cv::Point(pDet->x2, pDet->y2 - size.height),
                        cv::FONT_HERSHEY_COMPLEX, label_scale, color[0], box_think / 2, 8, 0);
        } 
        cv::imwrite("test_centernet.jpg", img);
    }


    /*
    HumanPoseResult* ptr = reinterpret_cast<HumanPoseResult*>(out_results[0].data()) ;
    int numhuman = out_results[0].size()*sizeof(float) / sizeof(HumanPoseResult);
    std::cout << " aa " << std::endl;
	std::vector<HumanPoseResult> re(ptr, ptr+numhuman);


	
	

    std::vector<cv::Scalar> color = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0)};
    //cv::Mat img = images[0].clone();
    drawResultImg(re, img, color);

    cv::imwrite("result_2b.jpg", img);

	return 0;
	auto en = eng.query_info("engine");
	if(!en.empty());
		SaveEngine("./model/multi_pose_dla_3x_static.engine", en);
	
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
    */
    return 0;
}
