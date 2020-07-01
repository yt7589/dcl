//#include "vehicle.h"
#include <pthread.h>
//#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include<vector>
//common
#include "file_util.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include "aes_api.hpp"
#include "vehicle_fgvc.h"
#include "predictor_api.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <codecvt>
#include <tuple>
#include "NvInferRuntimeCommon.h"
#define NUM_THREADS 1


#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "entropyCalibrator.h"


class Logger : public nvinfer1::ILogger           
 {
     void log(nvinfer1::ILogger::Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != nvinfer1::ILogger::Severity::kINFO)
             std::cout << msg << std::endl;
     }
 } gLogger;

const int DEPTH = 3;
const int IMG_W = 224;
const int IMG_H = 224;
int big_batchsize = 8;
int small_batchsize = 8;
const bool GPU_INPUT = true;
const bool GPU_DETECT_INPUT = true;
// 图片预处理相关，定义见config.py第17行
extern std::vector<float> g_rgb_mean; // = {0.485, 0.456, 0.406};
extern std::vector<float> g_rgb_std; // = {0.229, 0.224, 0.225};
extern double g_total_run_time;
extern int g_total_operation;
const float MEAN_R = g_rgb_mean[0] / g_rgb_std[0];
const float MEAN_G = g_rgb_mean[1] / g_rgb_std[1];
const float MEAN_B = g_rgb_mean[2] / g_rgb_std[2];
const float XFACTOR_R = 1.0 / (255.0 *  g_rgb_std[0]); // (g_rgb_std[0] * 255);
const float XFACTOR_G = 1.0 / (255.0 * g_rgb_std[0]); // 1 / (g_rgb_std[1] * 255);
const float XFACTOR_B = 1.0 / (255.0 * g_rgb_std[0]); // 1 / (g_rgb_std[2] * 255);

int FBLOCK_MAX_BYTES = 1024;
char *szBuf;
void Split(const std::string& src, const std::string& separator, std::vector<std::string>& dest);
vector<vector<string>> GetTestDsSamples();
int ProcessBatchImages(PredictorAPI* hand, std::vector<float> input_src, std::vector<cv::Mat> inputs, std::vector<int> results);

/**
 * 初始化检测模块，由于是单元测试，这里取每张图片中仅检出一辆车，而
 * 且该车为整张图片
 * 参数：
 *      det：车辆检测结果，其由iLeft,iTop, iRight, iBottom数组组成
 *          代表检测到车辆的位置
 *      detNum：检测到车辆的数量
 */
void  initDet(ITS_Vehicle_Result_Detect &det,const int detNum){
    det.CarNum = detNum;
    for(int i=0; i< detNum; ++i) {
        det.iLeft[i] =0;
        det.iTop[i] = 0;
        det.iRight[i] = IMG_W;
        det.iBottom[i] = IMG_H;
    }
}

/**
 * 从指定位置一次取批次大小（8）张图片
 * 参数：
 *      samples：二维数组，第一维是图片序号，第二维分别为图片文件
 *          路径和编号，例如：sample[i][0]代表第i张图片的文件路
 *          径，samples[i][1]为该图的类别编号
 *      startPos：开始位置，从0开始，第二次取时从8开始
 *      batchSize：批次大小，一次取图片数
 * 返回值：将图片缩小为224*224的图片列表（直接缩放未保持纵横比）
 */
std::tuple<std::vector<cv::Mat>, std::vector<int>> GetInputImage(vector<vector<string>> samples, int startPos, int batchSize)
{
    cv::Mat img;
    size_t imgSize;
    std::vector<cv::Mat> inputs;
    std::vector<int> results;
    for (int t = 0; t < batchSize; ++t)
    {
        img = cv::imread(samples[startPos + t][0], cv::IMREAD_COLOR);
        std::cout<<"img: "<<samples[startPos + t][0]<<"; classId: "<<samples[startPos + t][1]<<"; !!!!"<<std::endl;

        
        std::cout<<"### rows="<<img.rows<<"; cols="<<img.cols<<";"<<std::endl;
        for (int ii=0; ii<6; ii++)
        {
            cv::Vec3b pt = img.at<cv::Vec3b>(0, ii);
            std::cout<<"  "<<+static_cast<uint8_t>(pt[2])<<"  ";
        }
        std::cout<<std::endl;
        for (int ii=0; ii<6; ii++)
        {
            cv::Vec3b pt = img.at<cv::Vec3b>(ii, 0);
            std::cout<<"  *"<<+static_cast<uint8_t>(pt[2])<<"*  ";
        }



        cv::Mat resized;
        cv::resize(img, resized, cv::Size(IMG_W, IMG_H), 0, 0, cv::INTER_LINEAR);

        for (int ii=0; ii<6; ii++)
        {
            cv::Vec3b pt = resized.at<cv::Vec3b>(0, ii);
            std::cout<<"  ("<<+static_cast<uint8_t>(pt[2])<<")  ";
        }
        std::cout<<std::endl;
        
        std::cout<<std::endl;
        inputs.push_back(resized.clone());
        results.push_back(std::stoi(samples[startPos + t][1]));
    }
    std::tuple<std::vector<cv::Mat>, std::vector<int>> rst = 
            std::make_tuple(inputs, results);
    return rst;
}

/**
 * 由于在DCL训练中采用的是RGB格式图片，而在OpenCV中是BGR，所以需要做
 * 一下颜色通道顺序转换，同时在DCL中数值为0~1，这里需要还原为0~255
 * 参数：
 *      images：一个批次的8张图片
 */
std::vector<float> PreProcess(const std::vector<cv::Mat> &images)
{
    auto batch = images.size();
    auto depth = DEPTH;
    auto height = IMG_H;
    auto width = IMG_W;
    // 批处理识别
    std::vector<cv::Mat> split_mats;
    std::vector<float> dataVec(batch * height * width * depth);
    float *dataPtr = dataVec.data();
    for (const auto &image : images)
    {
        assert(image.rows == height);
        assert(image.cols == width);
        assert(image.type() == CV_8UC3 || image.type() == CV_32FC3);
        if (image.type() == CV_8UC3) 
        {
            std::cout<<"type: CV_8UC3"<<std::endl;
        }
        if (image.type() == CV_32FC3)
        {
            std::cout<<"type: CV_32FC3"<<std::endl;
        }
        std::vector<cv::Mat> channels;
        split(image, channels);
        cv::Mat imageRed(height, width, CV_32FC1, dataPtr);
        dataPtr += height * width;
        cv::Mat imageGreen(height, width, CV_32FC1, dataPtr);
        dataPtr += height * width;
        cv::Mat imageBlue(height, width, CV_32FC1, dataPtr);
        dataPtr += height * width;
        channels.at(0).convertTo(imageBlue, CV_32FC1, XFACTOR_B, -MEAN_B);
        channels.at(1).convertTo(imageGreen, CV_32FC1, XFACTOR_G, -MEAN_G);
        channels.at(2).convertTo(imageRed, CV_32FC1, XFACTOR_R, -MEAN_R);
        int basePos = 0;
        std::cout<<"dataVec_R: row: "<<dataVec[basePos]<<", "<<dataVec[basePos+1]<<", "<<dataVec[basePos+2]<<";col: "<<dataVec[basePos]<<", "<<dataVec[basePos+224]<<", "<<dataVec[basePos + 448]<<std::endl;
        basePos += height * width;
        std::cout<<"dataVec_G: row: "<<dataVec[basePos]<<", "<<dataVec[basePos+1]<<", "<<dataVec[basePos+2]<<";col: "<<dataVec[basePos]<<", "<<dataVec[basePos+224]<<", "<<dataVec[basePos + 448]<<std::endl;
        basePos += height * width;
        std::cout<<"dataVec_B: row: "<<dataVec[basePos]<<", "<<dataVec[basePos+1]<<", "<<dataVec[basePos+2]<<";col: "<<dataVec[basePos]<<", "<<dataVec[basePos+224]<<", "<<dataVec[basePos + 448]<<std::endl;
    }
    return dataVec;
}

/**
 * 将onnx文件转为TensorRT模型文件，支持int8量化
 * 参数：
 *      onnx_filename：onnx文件全路径文件名
 *      calibFilesTxt：用于int8量化标定的图像文件列表
 *      calibFilesPath：用于int8量化标定的图像文件目录
 *      trtFile：TensorRT模型文件
 * 使用示例见下面的call_convertOnnxToTrt
 */
void convertOnnxToTrt(const char* onnx_filename, const char* calibFilesTxt, 
            const char* calibFilesPath, const char* trtFile)
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    auto parser = nvonnxparser::createParser(*network, gLogger);
    std::cout<<"parser created"<<std::endl;
    bool rst = parser->parseFromFile(onnx_filename, 0);
    int maxBatchSize = 8;
    builder->setMaxBatchSize(maxBatchSize);
    std::cout<<"setMaxBatchSize is OK!"<<std::endl;
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::Int8EntropyCalibrator* calib = new nvinfer1::Int8EntropyCalibrator(
        maxBatchSize, calibFilesTxt, calibFilesPath, "cartyperec"
    );
    config->setInt8Calibrator(calib);
    if (builder->platformHasFastInt8()) 
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
    else
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    std::cout<<"create build config is OK"<<std::endl;
    config->setMaxWorkspaceSize(1 << 20);
    std::cout<<"setMaxWorkSpaceSize is OK"<<std::endl;
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("data", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3,224,224));
    profile->setDimensions("data", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(4, 3,224,224));
    profile->setDimensions("data", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(8, 3,224,224));
    config->addOptimizationProfile(profile);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout<<"buildEngineWithConfig is OK"<<std::endl;
    nvinfer1::IHostMemory *serializedModel = engine->serialize();
    std::cout<<"serialization of the engine! :"<<serializedModel<<"!"<<std::endl;
    std::ofstream ofs(trtFile, std::ios::out | std::ios::binary);
    ofs.write((char*)(serializedModel->data()), serializedModel->size());
    ofs.close();
    serializedModel->destroy();
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    std::cout<<"^_^ TensorRT ^_^"<<std::endl;
}
void call_convertOnnxToTrt()
{
    std::cout<<"将onnx模型转化为int8量化的trt文件 v0.0.1"<<std::endl;
    char* onnx_filename = "/media/zjkj/35196947-b671-441e-9631-6245942d671b/"
                            "yantao/fgvc/dcl/trt/vehicle_fgvc/models/dcl_v011.onnx";
    char* calibFilesTxt = "../models/calib_images.txt";
    char* calibFilesPath = "../models/images";
    char* trtFile = "../models/dcl_v011_int8_yt.trt";
    convertOnnxToTrt(onnx_filename, calibFilesTxt, calibFilesPath, trtFile);
}

void runTrtInfer(const char* modelfile)
{
    std::cout<<"loadTrtFile 1"<<std::endl;
    std::vector<char> trtModelStreamfromFile;
    size_t size{ 0 };
    std::ifstream file(modelfile, std::ios::binary);
    std::cout<<"loadTrtFile 2"<<std::endl;

    if (file.good())
    {
        std::cout<<"loadTrtFile 3"<<std::endl;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStreamfromFile.resize(size);
        file.read(trtModelStreamfromFile.data(), size);
        file.close();
        std::cout<<"loadTrtFile 4"<<std::endl;
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        std::cout<<"loadTrtFile 5"<<std::endl;
        nvinfer1::ICudaEngine* mEngine = runtime->deserializeCudaEngine(trtModelStreamfromFile.data(), size, nullptr);
        std::cout<<"loadTrtFile 6"<<std::endl;
        std::cout<<"loadTrtFile 7"<<std::endl;
        if (!mEngine)
        {
            std::cout<<"loadTrtFile 8"<<std::endl;
        }
        nvinfer1::IExecutionContext *context = mEngine->createExecutionContext();
        std::cout<<"create execution context is OK"<<std::endl;
        int inputIndex = mEngine->getBindingIndex("data");
        int outputIndex = mEngine->getBindingIndex("output");
        void* buffers[2];
        //buffers[inputIndex] = inputbuffer;
        //buffers[outputIndex] = outputBuffer;
        std::cout<<"推理过程成功！"<<std::endl;
    }
}
void call_runTrtInfer()
{
    char* modelfile = "/media/zjkj/35196947-b671-441e-9631-6245942d671b/"
                            "yantao/fgvc/dcl/trt/vehicle_fgvc/build/serialized_engine.trt";
    runTrtInfer(modelfile);
}

const int TEST_DS_NUM = 16; //5664; // 测试数据集记录数，必须能被8整除
static int init_num = 0;
void *mythread(void *threadid)
{
    int iDebug = 1;
    if (1 == iDebug)
    {
        //call_convertOnnxToTrt();
        call_runTrtInfer();
        return NULL;
    }
    int tid = *((int *)threadid);
    std::cout<<"TensorRT new int8 engine file test..."<<std::endl;
    /*std::string modelfile = "/media/zjkj/35196947-b671-441e-9631-6245942d671b/"
                            "yantao/fgvc/dcl/trt/vehicle_fgvc/models/dcl_v011_fp16.trt";*/
    std::string modelfile = "/media/zjkj/35196947-b671-441e-9631-6245942d671b/"
                            "yantao/fgvc/dcl/trt/vehicle_fgvc/build/serialized_engine.trt";
    auto hand = VehicleFgvcInstance(modelfile,
            tid % 4, small_batchsize, big_batchsize);
    std::cout<<"cropMain.mythread 1"<<std::endl;
    // 获取测试数据集上样本
    vector<vector<string>> samples = GetTestDsSamples();
    // Call other DCL interface
    int batchSize = 8;
    int startPos = 0;
    int correctNum = 0;
    int totalRecords = 0;
    for (startPos=8; startPos<TEST_DS_NUM; startPos+=8)
    {
        std::tuple<std::vector<cv::Mat>, std::vector<int>> rst = 
                    GetInputImage(samples, startPos, batchSize);
        auto inputs = std::get<0>(rst);
        auto targets = std::get<1>(rst);
        std::vector<float> input_src = PreProcess(inputs);
        std::cout<<"input_src: "<<input_src[0]<<", "<<input_src[1]<<", "<<input_src[2]<<"!"<<std::endl;
        correctNum += ProcessBatchImages((PredictorAPI*)hand, input_src, (std::vector<cv::Mat>)inputs, (std::vector<int>)targets);
        totalRecords += 8;
    }
    ReleaseVehicleFgvcInstance(hand);
    double ms_per_run = g_total_run_time / g_total_operation;
    std::cout<<"平均运行时间："<<ms_per_run<<"毫秒"<<std::endl;
    std::cout<<"准确率："<<(correctNum*1.0 / totalRecords)<<std::endl;
    return NULL;
}

int ProcessBatchImages(PredictorAPI* hand, std::vector<float> input_src, std::vector<cv::Mat> inputs, std::vector<int> targets)
{
    struct timeval start1;
    struct timeval end1;
    unsigned long timer;
    float *pGpu;
    void* deviceMem;
    cudaMalloc(&deviceMem, input_src.size() * sizeof(float));
    pGpu = (float*)deviceMem;
    cudaMemcpy(pGpu, input_src.data(),
                   input_src.size() * sizeof(float), cudaMemcpyHostToDevice);
    std::vector<cv::Mat> gpu_img ;
    std::vector<unsigned char*> cudaSrc;
    std::vector<int> srcWidth;
    std::vector<int> srcHeight;
    std::vector<ITS_Vehicle_Result_Detect> cpuDetect(inputs.size());
    for (int t = 0; t < inputs.size(); ++ t)
    {
        //cv::Mat m = cv::Mat::zeros(549, 549, CV_8UC3);
        cv::Mat m = cv::Mat::zeros(2*IMG_W, 2*IMG_H, CV_8UC3);
        cv::Rect rect(0, 0, IMG_W, IMG_H);
        cv::Rect rect2(IMG_W, IMG_H, IMG_W, IMG_H);  
        inputs[t].copyTo(m(rect));
        inputs[t].copyTo(m(rect2));
        int imgSize = m.step[0]*m.rows;
        auto & img = m;
        assert(imgSize = 3*img.cols*img.rows);
        void*pgpu;
        cudaMalloc((void**)&pgpu, imgSize);
        cudaSrc.push_back((unsigned char *)pgpu);
        cudaMemcpy(pgpu,img.data,imgSize,cudaMemcpyHostToDevice);
        srcWidth.push_back(img.cols);
        srcHeight.push_back(img.rows);
        initDet(cpuDetect[t], 1);
    }
    clock_t classify_start, classify_end;
    classify_start = clock();
    auto all_results = ClassifyVehicleFgvcFromDetectGPU(hand,
                                cudaSrc,srcWidth, srcHeight,
                                cpuDetect);
    classify_end = clock();
    //std::cout<<"程序运行时间："<<((double)(classify_end - classify_start))/CLOCKS_PER_SEC*1000.0<<"毫秒;"<<std::endl;
    int correctNum = 0;
    for (int u = 0; u < all_results.size(); ++u)
    {
        auto &RE = all_results[u];
        std::cout<<RE.tempResult[0].tempVehicleType<<"; classId="<<RE.tempResult[0].iVehicleSubModel<<"; target="<<targets[u]<<"!"<<std::endl;
        if (RE.tempResult[0].iVehicleSubModel == targets[u])
        {
            correctNum++;
        }
    }
    return correctNum;
}

/**
 * 单元测试程序入口，需要保证用于测试的测试数据集文件记录数能为8
 * 整除，如果不可以后面补足为8整除记录数
 */
int main()
{
    clock_t start, end;
    pthread_t threads[NUM_THREADS];
    int indexes[NUM_THREADS]; // 用数组来保存i的值
    for (int i = 0; i < NUM_THREADS; i++)
    {
        indexes[i] = i; //先保存i的值
        // 传入的时候必须强制转换为void* 类型，即无类型指针
        int rc = pthread_create(&threads[i], NULL, mythread, (void *)&(indexes[i]));
        if (rc)
        {
            std::cout << "Error:无法创建线程," << rc << std::endl;
            exit(-1);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
        //cout << "right:创建线程," << i << endl;
    }
    printf("end\n");
    //ITS_VehicleRecRelease(pInstance0);
    return 0;
}

vector<vector<string>> GetTestDsSamples()
{
    szBuf = (char*)malloc(FBLOCK_MAX_BYTES * sizeof(char) + 1);
    std::string strFileUTF8 = "/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/datasets/CUB_200_2011/anno/test_ds_v4.txt";
    std::string strFileName = strFileUTF8;
    memset(szBuf, 0, sizeof(char) * FBLOCK_MAX_BYTES);
    std::string strMessage;
    FILE * fp = NULL;
    fp = fopen(strFileName.c_str(), "rb");
    if (fp != NULL)
    {
        // fseek(fp, sizeof(char) * 3, 0);
        while(fread(szBuf, sizeof(char), FBLOCK_MAX_BYTES, fp) > 0)
        {
            szBuf[FBLOCK_MAX_BYTES] = '\0';
            strMessage += szBuf;
            memset(szBuf, 0, sizeof(char) * FBLOCK_MAX_BYTES);
        }
    }
    fclose(fp);
    vector<string> lines;
    vector<string> sample;
    vector<vector<string>> samples;
    Split(strMessage, "\n", lines);
    vector<string>::iterator iter;
    string line;
    for (iter=lines.begin(); iter!=lines.end(); iter++)
    {
        line = *iter;
        if (line.length() > 10) {
            Split(line, "*", sample);
            samples.push_back(sample);
        }
    }
    return samples;
}

void Split(const std::string& src, const std::string& separator, std::vector<std::string>& dest) //字符串分割到数组
{
 
        //参数1：要分割的字符串；参数2：作为分隔符的字符；参数3：存放分割后的字符串的vector向量
 
	string str = src;
	string substring;
	string::size_type start = 0, index;
	dest.clear();
	index = str.find_first_of(separator,start);
	do
	{
		if (index != string::npos)
		{    
			substring = str.substr(start,index-start );
			dest.push_back(substring);
			start =index+separator.size();
			index = str.find(separator,start);
			if (start == string::npos) break;
		}
	}while(index != string::npos);
 
	//the last part
	substring = str.substr(start);
	dest.push_back(substring);
}
