//
// Created by novio on 20-2-12.
//

#ifndef ONNX_RES50_ONNXTRT_H
#define ONNX_RES50_ONNXTRT_H


#include <NvOnnxParser.h>
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
                std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                std::vector<std::vector<CAR_FEATURE_RESULT>> &results);
        void infer(
                const std::vector<unsigned char *> &imgs,
                const std::vector<int> &heights,
                const std::vector<int> &widths,
                std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                std::vector<ITS_Vehicle_Result_Detect> &cpuPDet,
                std::vector<std::vector<CAR_FEATURE_RESULT>> &results);

    ~OnnxTRT() {
        cudaStreamDestroy(stream);
        cudaFree(cudaDet);
        cudaFree(cudaPDet);
        cudaFree(cudaBuffers[1]);
        free(cpuOutputbuffer);
        cudaFree(cudaImageBuffer);
    }

    private:
        int maxOriImageSize = 8;  //origin image batchsize
        int maxCarNum = 128;      //(origin image batchsize) * (carNum)
        int maxBatchSize;
        ITS_Vehicle_Result_Detect* cudaDet;
        ITS_Vehicle_Result_Detect* cudaPDet;
        vector<void *> cudaBuffers;
        vector<int > cudaBufferSize;
        void* cudaImageBuffer;
        void* cpuOutputbuffer;
        samplesCommon::OnnxSampleParams mParams;
        nvinfer1::Dims mInputDims{};
        nvinfer1::Dims mOutputDimsFeatures{};

        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
        OnnxUniquePtr<nvinfer1::IExecutionContext> context;
        cudaStream_t stream;
        int topK;

        bool
        constructNetwork(OnnxUniquePtr<nvinfer1::IBuilder> &builder,
                              OnnxUniquePtr<nvinfer1::INetworkDefinition> &network,
                              OnnxUniquePtr<nvinfer1::IBuilderConfig> &config,
                              OnnxUniquePtr<nvonnxparser::IParser> &parser);

    };
}


#endif //ONNX_RES50_ONNXTRT_H
