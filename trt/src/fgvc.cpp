#include <memory>

#include <memory>

#include <utility>

#include "fgvc.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/modules/core/include/opencv2/core/mat.hpp"
//#include <opencv2/imgcodecs.hpp>
#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/modules/imgcodecs/include/opencv2/imgcodecs.hpp"
//#include <opencv2/imgproc.hpp>
#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/modules/imgproc/include/opencv2/imgproc.hpp"
#include <chrono>
#include <unistd.h>
#include <iconv.h>
#include <common.h>
#include "sample_options.h"
#include "sample_engines.h"
#include "args_parser.h"
#include "buffers.h"
#include "parser_onnx_config.h"

#include "/home/zjkj/working_zjw/onnx--prog/TensorRT-6.0.1.5/include/NvInfer.h"
#include <cuda_runtime_api.h>
#include "batch_stream.h"
#include <cstdlib>
#include "entropy_calibrator.h"
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>
#include "onnx_trt.h"
#include "nv_crop_and_resize_novio.h"

JNADLL void *VehicleFeatureInstance(const string &modelPath, int cardNum,int max_batch_size)//端口初始化
{
    samplesCommon::OnnxSampleParams params;
    params.onnxFileName = modelPath+"dcl_yt2.onnx";
    params.inputTensorNames.emplace_back("data");
    params.batchSize = max_batch_size;
    params.outputTensorNames.emplace_back("features");
    params.gpuId = cardNum;
    params.engineFileName =modelPath+ "fgvc_"+std::to_string(max_batch_size)+".trt";
    params.dataDirs.emplace_back("/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/trt/calibrate_data");
    params.dataFile = "/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/trt/calibrate_data/calibrate_data_label.txt";
    params.int8 = true;
    cudaSetDevice(cardNum);
    std::cout << params.engineFileName << std::endl;
    CarFeatureExtract::OnnxTRT *handler = new CarFeatureExtract::OnnxTRT(params,0);
    handler->build(params.engineFileName);

    return handler;
}


JNADLL std::vector<std::vector<CAR_FEATURE_RESULT>> GetCarFeatureGPU(
        void *iInstanceId,
        const std::vector<unsigned char *> &imgs,
        const std::vector<int> &heights,
        const std::vector<int> &widths,
        std::vector<ITS_Vehicle_Result_Detect> &cpuDet) {
    std::vector<std::vector<CAR_FEATURE_RESULT>> toReturn{};
auto cudaDet = initTempCudaDet(cpuDet.size());
//std::cout<<"start copy cpu det to cuda"<<std::endl;
cudaMemcpy(cudaDet,cpuDet.data(), cpuDet.size()*sizeof(ITS_Vehicle_Result_Detect),cudaMemcpyHostToDevice);
//std::cout<<"finish copy cpu det to cuda"<<std::endl;

    ((CarFeatureExtract::OnnxTRT *) iInstanceId)->infer(imgs, heights, widths,cudaDet,cpuDet, toReturn);
freePointer(cudaDet);
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
    return std::vector<float>((float*)toParse.features,((float*)toParse.features)+toParse.featureNums);
}