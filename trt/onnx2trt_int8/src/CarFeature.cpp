#include <memory>
#include <utility>

#include "CarFeature.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <unistd.h>
#include <iconv.h>
#include <common.h>
#include "sampleOptions.h"
#include "sampleEngines.h"
#include "argsParser.h"
#include "buffers.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "BatchStream.h"
#include <cstdlib>
#include "entropyCalibrator.h"
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>
#include "OnnxTRT.h"
#include "nvCropAndResizeNovio.h"

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
    char* onnx_filename = "/hd10t/yantao/dcl/trt/onnx2trt_int8/models/dcl_v020.onnx";
    char* calibFilesTxt = "/hd10t/yantao/dcl/trt/vehicle_fgvc/models/calib_images.txt";
    char* calibFilesPath = "/hd10t/yantao/dcl/trt/vehicle_fgvc/models/images";
    char* trtFile = "/hd10t/yantao/dcl/trt/onnx2trt_int8/models/dcl_v011_int8_yt.trt";
    convertOnnxToTrt(onnx_filename, calibFilesTxt, calibFilesPath, trtFile);
}

JNADLL void *VehicleFeatureInstance(const string &modelPath, int cardNum,int max_batch_size)//端口初始化
{
    int iDebug = 10;
    if (1 == iDebug) 
    {
        call_convertOnnxToTrt();
        return NULL;
    }
    samplesCommon::OnnxSampleParams params;
    params.onnxFileName = modelPath + "dcl_pt12_v1.onnx";
    params.inputTensorNames.emplace_back("data");
    params.batchSize = max_batch_size;
    params.outputTensorNames.emplace_back("output");
    params.gpuId = cardNum;
    params.engineFileName =modelPath+ "dcl_pt12_q1.trt";
    params.dataDirs.emplace_back("");
    params.dataFile = "../models/calib_images.txt";
    params.int8 = true;
    std::cout<<"v0.0.1 CarFeature.VehicleFeatureInstance 1"<<std::endl;
    cudaSetDevice(cardNum);
    std::cout<<"CarFeature.VehicleFeatureInstance 2"<<std::endl;
    CarFeatureExtract::OnnxTRT *handler = new CarFeatureExtract::OnnxTRT(params,0);
    std::cout<<"CarFeature.VehicleFeatureInstance 3"<<std::endl;
    handler->build(params.engineFileName);
    std::cout<<"CarFeature.VehicleFeatureInstance 4"<<std::endl;

    return handler;
}


JNADLL std::vector<std::vector<CAR_FEATURE_RESULT>> GetCarFeatureGPU(
        void *iInstanceId,
        const std::vector<unsigned char *> &imgs,
        const std::vector<int> &heights,
        const std::vector<int> &widths,
        std::vector<ITS_Vehicle_Result_Detect> &cpuDet) {
      std::vector<std::vector<CAR_FEATURE_RESULT>> toReturn{};
      //int maxDetNum = cpuDet.size();
      //ITS_Vehicle_Result_Detect* cudadet;
      //cudaMalloc(&cudadet, maxDetNum * sizeof(ITS_Vehicle_Result_Detect));
      //cudaMemset(&cudadet,0,maxDetNum * sizeof(ITS_Vehicle_Result_Detect));
      //std::cout<<"finish copy cpu det to cuda"<<std::endl;

    ((CarFeatureExtract::OnnxTRT *) iInstanceId)->infer(imgs, heights, widths,cpuDet, toReturn);

    return std::move(toReturn);
}

JNADLL std::vector<std::vector<CAR_FEATURE_RESULT>> GetMaskedCarFeatureGPU(
        void *iInstanceId,
        const std::vector<unsigned char *> &imgs,
        const std::vector<int> &heights,
        const std::vector<int> &widths,
        std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
        std::vector<ITS_Vehicle_Result_Detect> &cpuPDet) {
    std::vector<std::vector<CAR_FEATURE_RESULT>> toReturn{};
    //int maxDetNum = cpuDet.size();
    //ITS_Vehicle_Result_Detect* cudadet;
    //cudaMalloc(&cudadet, maxDetNum * sizeof(ITS_Vehicle_Result_Detect));
    //cudaMemset(&cudadet,0,maxDetNum * sizeof(ITS_Vehicle_Result_Detect));
    //std::cout<<"finish copy cpu det to cuda"<<std::endl;

    ((CarFeatureExtract::OnnxTRT *) iInstanceId)->infer(imgs, heights, widths,cpuDet,cpuPDet, toReturn);

    return toReturn;
}

JNADLL int ReleaseSDKFeature(void *iInstanceId)//释放接口
{
    CarFeatureExtract::OnnxTRT *net = (CarFeatureExtract::OnnxTRT *) iInstanceId;
    if (NULL != net) {
        delete net;
    }
    return 0;
}

JNADLL std::vector<float> parseFEATUREVector(const CAR_FEATURE_RESULT &toParse){
    if(toParse.features == nullptr)
    {
	    return std::vector<float>(256,0);
    }
    return std::vector<float>((float*)toParse.features,((float*)toParse.features)+toParse.featureNums);
}
