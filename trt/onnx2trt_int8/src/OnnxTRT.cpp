//
// Created by novio on 20-2-12.
//


#include <OnnxTRT.h>
#include <string>
#include <NvInfer.h>
#include <logger.h>
#include <entropyCalibrator.h>
#include <sampleEngines.h>
#include "nvCropAndResizeNovio.h"
#include <cmath>
#include <cuda_runtime.h>
#include <common.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
//#include <cv.hpp>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorString(result),func);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
bool CarFeatureExtract::OnnxTRT::build(const std::string &engineFile) {
    if (!fileExist(engineFile)) {
        std::cout<<"src/OnnxTRT.cpp 1"<<std::endl;
        auto builder = OnnxTRT::OnnxUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
        if (!builder) {
            std::cout << "get builder failed!" << std::endl;
            return false;
        }
        std::cout<<"src/OnnxTRT.cpp 2"<<std::endl;
        auto network = OnnxUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
        if (!network) {
            std::cout << "get network failed!" << std::endl;
            return false;
        }
        std::cout<<"src/OnnxTRT.cpp 3"<<std::endl;
        auto config = OnnxUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cout << "get config failed!" << std::endl;
            return false;
        }
        std::cout<<"src/OnnxTRT.cpp 4"<<std::endl;
        auto parser = OnnxUniquePtr<nvonnxparser::IParser>(
                nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
        if (!parser) {
            std::cout << "get parser failed!" << std::endl;
            return false;
        }
        std::cout<<"src/OnnxTRT.cpp 5"<<std::endl;
        auto constructed = constructNetwork(builder, network, config, parser);
        if (!constructed) {
            std::cout << "get constructed failed!" << std::endl;
            return false;
        }
        //std::cout << engineFile << std::endl;
        std::cout<<"src/OnnxTRT.cpp 6"<<std::endl;

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config),
                                                         samplesCommon::InferDeleter());
        std::cout<<"src/OnnxTRT.cpp 7"<<std::endl;
        sample::saveEngine(*mEngine.get(), engineFile, std::cerr);
        std::cout<<"src/OnnxTRT.cpp 8"<<std::endl;
    } else {
        std::cout<<"src/OnnxTRT.cpp 9 engineFile="<<engineFile<<";"<<std::endl;
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(sample::loadEngine(engineFile, mParams.dlaCore, std::cerr),
                                                         samplesCommon::InferDeleter());
        std::cout<<"src/OnnxTRT.cpp 10"<<std::endl;
    }
    std::cout<<"src/OnnxTRT.cpp 11"<<std::endl;
    if (!mEngine) {
        return false;
    }
    context = OnnxUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    cudaStreamCreate(&stream);
    mInputDims = context->getBindingDimensions(0);
    assert(mInputDims.nbDims == 3);
    mOutputDimsFeatures = context->getBindingDimensions(1);
    cudaDet = initTempCudaDet(maxOriImageSize);
    cudaPDet = initTempCudaDet(maxOriImageSize);
    cudaBuffers.resize( mEngine->getNbBindings());
    cudaBufferSize.resize(mEngine->getNbBindings());
    for (int i = 0; i < mEngine->getNbBindings(); i++) {
        auto dims = mEngine->getBindingDimensions(i);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        cudaBufferSize[i] = samplesCommon::volume(dims) * samplesCommon::getElementSize(type);
    }
    maxBatchSize = mEngine->getMaxBatchSize();
    cudaImageBuffer = samplesCommon::safeCudaMalloc(cudaBufferSize[0] * maxCarNum);
    cudaBuffers[1] = samplesCommon::safeCudaMalloc(cudaBufferSize[1] * maxBatchSize );
    cpuOutputbuffer = malloc(cudaBufferSize[1] * maxBatchSize);

    return true;
}

bool CarFeatureExtract::OnnxTRT::constructNetwork(OnnxUniquePtr<nvinfer1::IBuilder> &builder,
                               OnnxUniquePtr<nvinfer1::INetworkDefinition> &network,
                               OnnxUniquePtr<nvinfer1::IBuilderConfig> &config,
                               OnnxUniquePtr<nvonnxparser::IParser> &parser) {
    std::cout << this->mParams.onnxFileName.c_str() << std::endl;
    auto parsed = parser->parseFromFile(
            this->mParams.onnxFileName.c_str(),
            static_cast<int>(gLogger.getReportableSeverity()));
    std::cout<<"src/OnnxTRT.cpp.constructNetwork 1"<<std::endl;
    if (!parsed) {
        std::cout << "feature extraction get parsed failed!" << std::endl;
        return false;
    }
    builder->setMaxBatchSize(mParams.batchSize);
    config->setFlag(BuilderFlag::kDEBUG);
    config->setMaxWorkspaceSize(1024_MiB);
    if (mParams.fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8 && builder->platformHasFastInt8()) {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(
                new Int8EntropyCalibrator(mParams.calBatchSize, mParams.dataFile, mParams.dataDirs[0], "cartyperec"));
    }
    return true;
}
void  l2norm(float *values,const float* source_v,int valueCount){
    float accumulate = 0.f;
    for(int i = 0;i < valueCount;++i){
        accumulate += source_v[i] * source_v[i];
    }
    for(int i = 0;i < valueCount;++i){
        values[i] = source_v[i] / accumulate;
    }
}
void CarFeatureExtract::OnnxTRT::infer(
        const std::vector<unsigned char *> &imgs,
        const std::vector<int> &heights,
        const std::vector<int> &widths,
        std::vector<ITS_Vehicle_Result_Detect>& cpuDet,
        std::vector<std::vector<CAR_FEATURE_RESULT>> &results) {

    int batchSize = imgs.size();
    results.resize(batchSize);
    float means[] = {0.406f, 0.456f, 0.485f};
    float stds[] = {0.225f, 0.224f, 0.229f};
    CUDA_CHECK(cudaMemcpy(cudaDet,cpuDet.data(),
                     cpuDet.size()*sizeof(ITS_Vehicle_Result_Detect),cudaMemcpyHostToDevice));
    int carNum = nvCropAndResizeAndNormLaunch((float*)cudaImageBuffer, imgs, cudaDet,cpuDet, widths, heights,
                                 maxCarNum, 384, 384,means, stds);
    // 进行模型的推断
    int numBatch = carNum / maxBatchSize;
    int numLastCarPic= carNum % maxBatchSize;
    int oriImgId = 0,curCarnum = 0;
    for ( int i = 0; i < numBatch ; ++i){
        cudaBuffers[0] = static_cast<char *>(cudaImageBuffer) + i*maxBatchSize*cudaBufferSize[0];
        bool status = context->execute(maxBatchSize, cudaBuffers.data());
        CUDA_CHECK(cudaMemcpy(cpuOutputbuffer, cudaBuffers[1],
                         cudaBufferSize[1]*maxBatchSize,cudaMemcpyDeviceToHost));
        for(int j=0; j < maxBatchSize; ++j){
            if(results[oriImgId].size() != cpuDet[oriImgId].CarNum)
                results[oriImgId].resize(cpuDet[oriImgId].CarNum);
            l2norm(reinterpret_cast<float *>(results[oriImgId][curCarnum].features),
                   static_cast<float *>(cpuOutputbuffer) + j*mOutputDimsFeatures.d[0]
                    ,mOutputDimsFeatures.d[0]);
            results[oriImgId][curCarnum].confidence = 0;
            results[oriImgId][curCarnum].PINPAI = 0;
            results[oriImgId][curCarnum].featureNums = mOutputDimsFeatures.d[0];
            curCarnum++;
            if(curCarnum == cpuDet[oriImgId].CarNum){
                oriImgId++;
                curCarnum = 0;
            }

        }
    }
    if(numLastCarPic > 0){
        cudaBuffers[0] = static_cast<char *>(cudaImageBuffer) + numBatch*maxBatchSize*cudaBufferSize[0];
        bool status = context->execute(numLastCarPic, cudaBuffers.data());
        CUDA_CHECK(cudaMemcpy(cpuOutputbuffer, cudaBuffers[1],
                         cudaBufferSize[1]*numLastCarPic,cudaMemcpyDeviceToHost));
        for(int j=0; j < numLastCarPic; ++j){
            if(results[oriImgId].size() != cpuDet[oriImgId].CarNum)
                results[oriImgId].resize(cpuDet[oriImgId].CarNum);
            l2norm(reinterpret_cast<float *>(results[oriImgId][curCarnum].features),
                   static_cast<float *>(cpuOutputbuffer) + j*mOutputDimsFeatures.d[0]
                    ,mOutputDimsFeatures.d[0]);
            results[oriImgId][curCarnum].confidence = 0;
            results[oriImgId][curCarnum].PINPAI = 0;
            results[oriImgId][curCarnum].featureNums = mOutputDimsFeatures.d[0];
            curCarnum++;
            if(curCarnum == cpuDet[oriImgId].CarNum){
                oriImgId++;
                curCarnum = 0;
            }
        }
    }
}


void CarFeatureExtract::OnnxTRT::infer(
        const std::vector<unsigned char *> &imgs,
        const std::vector<int> &heights,
        const std::vector<int> &widths,
        std::vector<ITS_Vehicle_Result_Detect>& cpuDet,
        std::vector<ITS_Vehicle_Result_Detect> &cpuPDet,
        std::vector<std::vector<CAR_FEATURE_RESULT>> &results) {

    int batchSize = imgs.size();
    results.resize(batchSize);
    float means[] = {0.406f, 0.456f, 0.485f};
    float stds[] = {0.225f, 0.224f, 0.229f};
    cudaMemcpy(cudaDet,cpuDet.data(), cpuDet.size()*sizeof(ITS_Vehicle_Result_Detect),cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPDet,cpuPDet.data(), cpuPDet.size()*sizeof(ITS_Vehicle_Result_Detect),cudaMemcpyHostToDevice);
    int carNum = nvCropAndResizeAndNormLaunch((float*)cudaImageBuffer, imgs,
            cudaDet,cudaPDet,cpuDet, widths, heights,
            maxCarNum, 384, 384,means, stds);

    int numBatch = carNum / maxBatchSize;
    int numLastCarPic= carNum % maxBatchSize;
    int oriImgId = 0,curCarnum = 0;
    for ( int i = 0; i< numBatch ; ++i){
        cudaBuffers[0] = static_cast<char *>(cudaImageBuffer) + i*maxBatchSize*cudaBufferSize[0];
        bool status = context->execute(maxBatchSize, cudaBuffers.data());
        CUDA_CHECK(cudaMemcpy(cpuOutputbuffer, cudaBuffers[1],
                         cudaBufferSize[1]*maxBatchSize,cudaMemcpyDeviceToHost));
        for(int j=0; j < maxBatchSize; ++j){
            if(results[oriImgId].size() != cpuDet[oriImgId].CarNum)
                results[oriImgId].resize(cpuDet[oriImgId].CarNum);
            l2norm(reinterpret_cast<float *>(results[oriImgId][curCarnum].features),
                   static_cast<float *>(cpuOutputbuffer) + j*mOutputDimsFeatures.d[0]
                    ,mOutputDimsFeatures.d[0]);
            results[oriImgId][curCarnum].confidence = 0;
            results[oriImgId][curCarnum].PINPAI = 0;
            results[oriImgId][curCarnum].featureNums = mOutputDimsFeatures.d[0];
            curCarnum++;
            if(curCarnum == cpuDet[oriImgId].CarNum){
                oriImgId++;
                curCarnum = 0;
            }

        }
    }
    if(numLastCarPic > 0){
        cudaBuffers[0] = static_cast<char *>(cudaImageBuffer) + numBatch*maxBatchSize*cudaBufferSize[0];
        bool status = context->execute(numLastCarPic, cudaBuffers.data());
        CUDA_CHECK(cudaMemcpy(cpuOutputbuffer, cudaBuffers[1],
                         cudaBufferSize[1]*numLastCarPic,cudaMemcpyDeviceToHost));
        for(int j=0; j < numLastCarPic; ++j){
            if(results[oriImgId].size() != cpuDet[oriImgId].CarNum)
                results[oriImgId].resize(cpuDet[oriImgId].CarNum);
            l2norm(reinterpret_cast<float *>(results[oriImgId][curCarnum].features),
                   static_cast<float *>(cpuOutputbuffer) + j*mOutputDimsFeatures.d[0]
                    ,mOutputDimsFeatures.d[0]);
            results[oriImgId][curCarnum].confidence = 0;
            results[oriImgId][curCarnum].PINPAI = 0;
            results[oriImgId][curCarnum].featureNums = mOutputDimsFeatures.d[0];
            curCarnum++;
            if(curCarnum == cpuDet[oriImgId].CarNum){
                oriImgId++;
                curCarnum = 0;
            }
        }

}
}
