//
// Created by novio on 20-2-12.
//

#ifndef __ONNX_TRT_H__
#define __ONNX_TRT_H__


#include "/home/zjkj/working_zjw/onnx--prog/TensorRT-6.0.1.5/include/NvOnnxParser.h"
#include "common.h"
#include "argsParser.h"
#include "CarFeature.h"
#include "buffers.h"
#include "sampleOptions.h"

namespace CarFeatureExtract{
    class OnnxTRT {
        template<typename T>
        using OnnxUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

    public:
        OnnxTRT(samplesCommon::OnnxSampleParams params, int topK)
                : mParams(std::move(params)), mEngine(nullptr), topK(topK) {
        }

        //!
        //! \brief Function builds the network engine
        //!
        bool build(const std::string &);

        //!
        //! \brief Runs the TensorRT inference engine for this sample
        //!

        void infer(
                const std::vector<unsigned char *> &imgs,
                const std::vector<int> &heights,
                const std::vector<int> &widths,
                ITS_Vehicle_Result_Detect* cudaDet,
std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                std::vector<std::vector<CAR_FEATURE_RESULT>> &results);

//    ~OnnxTRT() {
//        cudaStreamDestroy(stream);
//        context->destroy();
//    }

    private:
        samplesCommon::OnnxSampleParams mParams;
        nvinfer1::Dims mInputDims{};
        nvinfer1::Dims mOutputDimsFeatures{};

        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
        OnnxUniquePtr<nvinfer1::IExecutionContext> context;
        cudaStream_t stream;
        int topK;

        bool constructNetwork(OnnxUniquePtr<nvinfer1::IBuilder> &builder,
                              OnnxUniquePtr<nvinfer1::INetworkDefinition> &network,
                              OnnxUniquePtr<nvinfer1::IBuilderConfig> &config,
                              OnnxUniquePtr<nvonnxparser::IParser> &parser);


        bool
        processInput(
                const std::vector<unsigned char *> &imgs,
                const std::vector<int> &heights,
                const std::vector<int> &widths,
                ITS_Vehicle_Result_Detect *cudaDet,
std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                const samplesCommon::BufferManager &buffers);

        bool
        verifyOutput(const samplesCommon::BufferManager &buffers, int batchSize,
                     std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                     std::vector<std::vector<CAR_FEATURE_RESULT>> &results);
    };
}


#endif // __ONNX_TRT_H__