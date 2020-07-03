#include <utility>

//
// Created by cao on 19-12-16.
//

#include "entropyCalibrator.h"
#include <fstream>
#include <iterator>
#include <iostream>
#include <cuda_runtime.h>
#include <common.h>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace nvinfer1 {

    Int8EntropyCalibrator::Int8EntropyCalibrator(
            const int &batchSize,
            const std::string &dataFile,
            const std::string &imgFolder,
            std::string calibrationTable) : batchSize(batchSize),
                                            calibrationTable(std::move(calibrationTable)),
                                            imageIndex(0) {
        int inputChannel = 3;
        int inputH = 224;
        int inputW = 224;
        inputCount = batchSize * inputChannel * inputH * inputW;
        std::fstream f(dataFile);
        if (f.is_open()) {
            std::string temp, word;
            std::vector<std::string> rowValue{};
            while (std::getline(f, temp)) {
                std::stringstream s(temp);
                rowValue.clear();
                while (getline(s, word, ',')) {
                    rowValue.push_back(word);
                }
                imgPaths.emplace_back('/' + rowValue[0]);
                if(imgPaths.size()>950){break;}

            }
        }
        // 保证不会存在小于batchsize的batch
        imgPaths.resize((imgPaths.size()/batchSize)*batchSize);
        batchData = new float[inputCount];
        CHECK(cudaMalloc(&deviceInput, inputCount * sizeof(float)));
    }

    Int8EntropyCalibrator::~Int8EntropyCalibrator() {
        CHECK(cudaFree(deviceInput));
        if (batchData) {
            delete[] batchData;
        }
    }

    bool Int8EntropyCalibrator::getBatch(void **bindings, const char **names, int nbBindings) {
        if (imageIndex + batchSize > int(imgPaths.size()))
            return false;
        // load batch
        float *ptr = batchData;
        //float means[] = {0.406f, 0.456f, 0.485f};
        //float stds[] = {0.225f, 0.224f, 0.229f};
	float means[] = {0.485f, 0.485f, 0.485f};
	float stds[] = {0.225f, 0.225f, 0.225f};
        int perChannelPixels = 224 * 224; // feature extractor is 384
        int perImagePixels = perChannelPixels * 3; 
        std::vector<cv::Mat> bgrImages{};
        int startIndex = 0;
        for (size_t j = imageIndex; j < imageIndex + batchSize; ++j) {
            bgrImages.clear();
            auto img = cv::imread(imgPaths[j]);
	    //auto img = cv::imread("/hd10t/yantao/dcl/trt/onnx2trt_int8/models/images/img_00001.jpg");
            std::cout << imgPaths[j] << std::endl;
            cv::Mat img_dst;
            cv::resize(img, img_dst, cv::Size(224, 224)); // feature extractor is 384
            cv::split(img_dst, bgrImages);
	    /*
            for (int k = 0; k < 3; ++k) {
                startIndex = k * perChannelPixels + perImagePixels * (j-imageIndex);
                std::vector<uint8_t> imgValues{bgrImages[k].data, bgrImages[k].data + perChannelPixels};
                std::transform(imgValues.begin(), imgValues.end(),
                               ptr + startIndex,
                               [&](uint8_t val) { return (static_cast<float>(val) / 255.f - means[k]) / stds[k]; });
            }
	    */
	    for (int k = 0; k < 3; ++k) {
                startIndex = k * perChannelPixels + perImagePixels * (j-imageIndex);
                std::vector<uint8_t> imgValues{bgrImages[2-k].data, bgrImages[2-k].data + perChannelPixels};
                std::transform(imgValues.begin(), imgValues.end(),
                               ptr + startIndex,
                               [&](uint8_t val) { return (static_cast<float>(val) / 255.f - means[k]) / stds[k]; });
            }

        }
        imageIndex += batchSize;
        CHECK(cudaMemcpy(deviceInput, batchData, inputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[0] = deviceInput;
        return true;
    }

    const void *Int8EntropyCalibrator::readCalibrationCache(std::size_t &length) {
        calibrationCache.clear();
        std::ifstream input(calibrationTable, std::ios::binary);
        input >> std::noskipws;
        if (readCache && input.good() && false)
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(calibrationCache));

        length = calibrationCache.size();
        return length ? &calibrationCache[0] : nullptr;
    }

    void Int8EntropyCalibrator::writeCalibrationCache(const void *cache, std::size_t length) {
        std::ofstream output(calibrationTable, std::ios::binary);
        output.write(reinterpret_cast<const char *>(cache), length);
    }

    CalibrationAlgoType Int8EntropyCalibrator::getAlgorithm() {
        return CalibrationAlgoType::kENTROPY_CALIBRATION;
    }

}
