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

JNADLL void *VehicleFeatureInstance(const string &modelPath, int cardNum,int max_batch_size)//端口初始化
{
    samplesCommon::OnnxSampleParams params;
    params.onnxFileName = modelPath + "dcl_v006_sim.onnx";
    params.inputTensorNames.emplace_back("data");
    params.batchSize = max_batch_size;
    params.outputTensorNames.emplace_back("output");
    params.gpuId = cardNum;
    params.engineFileName =modelPath+ "dcl_v005_q.trt";
    params.dataDirs.emplace_back("");
    params.dataFile = "../models/calib_images_all.txt";
    params.int8 = true;
    std::cout<<"CarFeature.VehicleFeatureInstance 1"<<std::endl;
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
