//
// Created by novio on 20-2-12.
//


#include "onnx_trt.h"
#include <string>
#include "/home/zjkj/working_zjw/onnx--prog/TensorRT-6.0.1.5/include/NvInfer.h"
#include "logger.h"
#include "entropy_calibrator.h"
#include "sample_engines.h"
#include "nv_crop_and_resize_novio.h"
#include <cmath>

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
    auto builder = OnnxTRT::OnnxUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }

    auto network = OnnxUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) {
        return false;
    }

    auto config = OnnxUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    auto parser = OnnxUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed) {
        return false;
    }
    if (!fileExist(engineFile)) {
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config),
                                                         samplesCommon::InferDeleter());
        sample::saveEngine(*mEngine.get(), engineFile, std::cerr);
    } else {
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(sample::loadEngine(engineFile, mParams.dlaCore, std::cerr),
                                                         samplesCommon::InferDeleter());
    }
    if (!mEngine) {
        return false;
    }
    context = OnnxUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    cudaStreamCreate(&stream);
    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    assert(network->getNbOutputs() == 1);
    mOutputDimsFeatures = network->getOutput(0)->getDimensions();

    return true;
}

bool CarFeatureExtract::OnnxTRT::constructNetwork(OnnxUniquePtr<nvinfer1::IBuilder> &builder,
                               OnnxUniquePtr<nvinfer1::INetworkDefinition> &network,
                               OnnxUniquePtr<nvinfer1::IBuilderConfig> &config,
                               OnnxUniquePtr<nvonnxparser::IParser> &parser) {
    auto parsed = parser->parseFromFile(
            this->mParams.onnxFileName.c_str(),
            static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
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

void CarFeatureExtract::OnnxTRT::infer(
        const std::vector<unsigned char *> &imgs,
        const std::vector<int> &heights,
        const std::vector<int> &widths,
        ITS_Vehicle_Result_Detect *cudaDet,
std::vector<ITS_Vehicle_Result_Detect>& cpuDet,
        std::vector<std::vector<CAR_FEATURE_RESULT>> &results) {
    assert(imgs.size() <= mParams.batchSize);
    int allImages = 0;
for(auto &mDet : cpuDet){allImages += mDet.CarNum;}
    // 创建内存空间用于加载需要进行预测的数据
    std::cout << mEngine << std::endl;
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
    if (!context) {
        return;
    }

    // 对数据进行初始化
    if (!processInput(imgs, heights, widths, cudaDet,cpuDet, buffers)) {
        return;
    }

    // 进行模型的推断
    bool status = context->execute(allImages, buffers.getDeviceBindings().data());
    if (!status) {
        return;
    }
    // 将显存中的结果导出到内存
    buffers.copyOutputToHostAsync(stream);
    cudaStreamSynchronize(stream);

    // 验证结果并进行封装
    verifyOutput(buffers, allImages,cpuDet, results);

}

bool CarFeatureExtract::OnnxTRT::processInput(
        const std::vector<unsigned char *> &imgs,
        const std::vector<int> &heights,
        const std::vector<int> &widths,
        ITS_Vehicle_Result_Detect* cudaDet,
std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
        const samplesCommon::BufferManager &buffers) {
    assert (imgs.size() == heights.size() && imgs.size() == widths.size());
    float means[] = {0.406f, 0.456f, 0.485f};
    float stds[] = {0.225f, 0.224f, 0.229f};
    auto *deviceDataBuffer = static_cast<float *>(buffers.getDeviceBuffer(mParams.inputTensorNames[0]));

    nvCropAndResizeAndNormLaunch(deviceDataBuffer, imgs, cudaDet,cpuDet, widths, heights, mParams.batchSize, 224, 224,means, stds);

    return true;
}

float* l2norm(const float *values,int valueCount){
    auto* toReturn = new float[valueCount];
    memset(toReturn,0,sizeof(float)*valueCount);
    float accumulate{0.0};
    for(int i = 0;i < valueCount;++i){
        accumulate += values[i] * values[i];
    }
    for(int i = 0;i < valueCount;++i){
        toReturn[i] = values[i] / accumulate;
    }
    return toReturn;
}

bool CarFeatureExtract::OnnxTRT::verifyOutput(const samplesCommon::BufferManager &buffers, int batchSize,
                           std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                           std::vector<std::vector<CAR_FEATURE_RESULT>> &results) {
    const int featureSize = mOutputDimsFeatures.d[0];
    float *features = static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    std::vector<CAR_FEATURE_RESULT> tempResults{};
    for (int j = 0; j < batchSize; ++j) {
        CAR_FEATURE_RESULT result;
        const int mBatchFeatureIndex = j * featureSize;
        result.confidence = 0;
        result.PINPAI = 0;
        result.featureNums = featureSize;
        auto normedFeature = l2norm(features+mBatchFeatureIndex,featureSize);
        memset(result.features,0,sizeof(result.features));
        memcpy(result.features,normedFeature,sizeof(float)*featureSize);
        tempResults.emplace_back(result);
        free(normedFeature);
    }

//std::cout<<"total results:"<<tempResults.size()<<std::endl;
    int totalSum = 0;
    for(auto& mDet : cpuDet){
        std::vector<CAR_FEATURE_RESULT> mResult{};
        for(int i = 0;i < mDet.CarNum;++i){
//std::cout<<"start emplace index:"<<totalSum+i<<std::endl;
            mResult.emplace_back(tempResults[totalSum+i]);
        }
        totalSum += mDet.CarNum;
        results.emplace_back(mResult);
    }

    return true;
}