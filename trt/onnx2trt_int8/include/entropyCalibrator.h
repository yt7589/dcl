//
// Created by cao on 19-12-16.
//

#ifndef entropyCalibrator
#define entropyCalibrator

#include "NvInfer.h"
#include <vector>
#include <string>

namespace nvinfer1 {
    class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator {
    public:
        Int8EntropyCalibrator(const int &batchSize,
                              const std::string &dataFile, const std::string &imgFolder,
                              std::string calibrationTable);

        virtual ~Int8EntropyCalibrator();

        int getBatchSize() const override { return batchSize; }

        bool getBatch(void *bindings[], const char *names[], int nbBindings) override;

        const void *readCalibrationCache(std::size_t &length) override;

        void writeCalibrationCache(const void *ptr, std::size_t length) override;
        CalibrationAlgoType getAlgorithm() override;
    private:

        int batchSize;
        size_t inputCount;
        size_t imageIndex;

        std::string calibrationTable;
        std::vector<std::string> imgPaths{};

        float *batchData{nullptr};
        void *deviceInput{nullptr};


        bool readCache;
        std::vector<char> calibrationCache;
    };
}


#endif
