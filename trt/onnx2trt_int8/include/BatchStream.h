/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include "common.h"
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>

class IBatchStream {
public:
    virtual void reset(int firstBatch) = 0;

    virtual bool next() = 0;

    virtual void skip(int skipCount) = 0;

    virtual float *getBatch() = 0;

    virtual float *getLabels() = 0;

    virtual int getBatchesRead() const = 0;

    virtual int getBatchSize() const = 0;

    virtual nvinfer1::Dims getDims() const = 0;

    virtual nvinfer1::Dims getImageDims() const = 0;
};

class VehicleBatchStream : public IBatchStream {
public:
    /**
     * 加载标定用数据流
     * @param batchSize 批处理大小
     * @param maxBatches 训练的批次数
     * @param dataListFile 数据文件路径
     * @param directories 图像数据的目标路径
     */
    VehicleBatchStream(int batchSize,
                       const std::string &dataListFile,
                       const std::string &directories)
            : mBatchSize{batchSize},mMaxBatches{100} {
        mDims = Dims3(3, 224, 224);
        loadCalibrationData(directories,dataListFile);
    }

    void reset(int firstBatch) override {
        mBatchCount = firstBatch;
    }

    bool next() override {
        if (mBatchCount >= mMaxBatches) {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override {
        mBatchCount += skipCount;
    }

    float *getBatch() override {
        return mData.data() + (mBatchCount * mBatchSize * samplesCommon::volume(mDims));
    }

    float *getLabels() override {
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override {
        return mBatchCount;
    }

    int getBatchSize() const override {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override {
        return mDims;
    }

    nvinfer1::Dims getImageDims() const override {
        return Dims3{3, 224, 224};
    }

private:
    int mBatchSize{0};
    int mBatchCount{0}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};

    void loadCalibrationData(const std::string &imgDataDirectory, const std::string &dataListFile) {
        std::ifstream toReadFile{dataListFile.c_str(), std::ios::in};
        string line, word, temp;
        std::vector<std::string> rowValue{};
        float means[] = {0.406f, 0.456f, 0.485f};
        float stds[] = {0.225f, 0.224f, 0.229f};
        int imageCount = 0;
        int perImagePixels = sample::volume(mDims);
        int perChannelPixels = mDims.d[1] * mDims.d[2];
        std::vector<cv::Mat> bgrImages{};
        int startIndex;
        int maxImages = mBatchSize*mMaxBatches;
        mData.resize(maxImages*perImagePixels);
        while (getline(toReadFile, line) && imageCount < maxImages) {
            if(line.empty()){
                continue;
            }
            stringstream s(line);
            while (getline(s, word, ',')) {
                rowValue.push_back(word);
            }
            cv::Mat mImage = cv::imread(imgDataDirectory + '/' + rowValue[0]);
            cv::resize(mImage,mImage,cv::Size(224,224));
//            std::cout<<mImage.size<<std::endl;
            cv::split(mImage, bgrImages);
            // 将每个维度的数据进行处理，并且加到全局数据中
            // 图像已经预先处理成为224的图像了
            for (int j = 0; j < mDims.d[0]; ++j) {
                startIndex = j * perChannelPixels + perImagePixels * imageCount;
                std::vector<uint8_t > imgValues{bgrImages[j].data, bgrImages[j].data + perChannelPixels};
                std::transform(imgValues.begin(),imgValues.end(),
                               mData.begin() + startIndex,
                               [&](uint8_t val) { return (static_cast<float>(val) / 255.f - means[j]) / stds[j]; });
            }
            mLabels.emplace_back(static_cast<float>(std::stof(rowValue[1])));
            ++imageCount;
            bgrImages.clear();
            rowValue.clear();
        }
    }

};

class BatchStream : public IBatchStream {
public:
    BatchStream(
            int batchSize, int maxBatches, std::string prefix, std::string suffix, std::vector<std::string> directories)
            : mBatchSize(batchSize), mMaxBatches(maxBatches), mPrefix(prefix), mSuffix(suffix), mDataDir(directories) {
        FILE *file = fopen(locateFile(mPrefix + std::string("0") + mSuffix, mDataDir).c_str(), "rb");
        assert(file != nullptr);
        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        mDims.nbDims = 4;  // The number of dimensions.
        mDims.d[0] = d[0]; // Batch Size
        mDims.d[1] = d[1]; // Channels
        mDims.d[2] = d[2]; // Height
        mDims.d[3] = d[3]; // Width
        assert(mDims.d[0] > 0 && mDims.d[1] > 0 && mDims.d[2] > 0 && mDims.d[3] > 0);
        fclose(file);

        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    BatchStream(int batchSize, int maxBatches, std::string prefix, std::vector<std::string> directories)
            : BatchStream(batchSize, maxBatches, prefix, ".batch", directories) {
    }

    // This constructor expects that the dimensions include the batch dimension.
    BatchStream(int maxBatches, nvinfer1::Dims dims, std::string listFile, std::vector<std::string> directories)
            : mBatchSize(dims.d[0]), mMaxBatches(maxBatches), mDims(dims), mListFile(listFile), mDataDir(directories) {
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    // Resets data members
    void reset(int firstBatch) override {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next() override {
        if (mBatchCount == mMaxBatches) {
            return false;
        }

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize) {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
            if (mFileBatchPos == mDims.d[0] && !update()) {
                return false;
            }

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(
                    getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize,
                    getBatch() + batchPos * mImageSize);
            std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
        }
        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount) override {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0]) {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++) {
            next();
        }
        mBatchCount = x;
    }

    float *getBatch() override {
        return mBatch.data();
    }

    float *getLabels() override {
        return mLabels.data();
    }

    int getBatchesRead() const override {
        return mBatchCount;
    }

    int getBatchSize() const override {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override {
        return mDims;
    }

    nvinfer1::Dims getImageDims() const override {
        return Dims3{mDims.d[1], mDims.d[2], mDims.d[3]};
    }

private:
    float *getFileBatch() {
        return mFileBatch.data();
    }

    float *getFileLabels() { return mFileLabels.data(); }

    bool update() {
        if (mListFile.empty()) {
            std::string inputFileName = locateFile(mPrefix + std::to_string(mFileCount++) + mSuffix, mDataDir);
            FILE *file = fopen(inputFileName.c_str(), "rb");
            if (!file) {
                return false;
            }

            int d[4];
            size_t readSize = fread(d, sizeof(int), 4, file);
            assert(readSize == 4);
            assert(mDims.d[0] == d[0] && mDims.d[1] == d[1] && mDims.d[2] == d[2] && mDims.d[3] == d[3]);
            size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
            assert(readInputCount == size_t(mDims.d[0] * mImageSize));
            size_t readLabelCount = fread(getFileLabels(), sizeof(float), mDims.d[0], file);
            assert(readLabelCount == 0 || readLabelCount == size_t(mDims.d[0]));

            fclose(file);
        } else {
            std::vector<std::string> fNames;
            std::ifstream file(locateFile(mListFile, mDataDir), std::ios::binary);
            if (!file) {
                return false;
            }

            gLogInfo << "Batch #" << mFileCount << std::endl;
            file.seekg(((mBatchCount * mBatchSize)) * 7);

            for (int i = 1; i <= mBatchSize; i++) {
                std::string sName;
                std::getline(file, sName);
                sName = sName + ".ppm";
                gLogInfo << "Calibrating with file " << sName << std::endl;
                fNames.emplace_back(sName);
            }

            mFileCount++;

            const int imageC = 3;
            const int imageH = 300;
            const int imageW = 300;
            std::vector<samplesCommon::PPM<imageC, imageH, imageW>> ppms(fNames.size());
            for (uint32_t i = 0; i < fNames.size(); ++i) {
                readPPMFile(locateFile(fNames[i], mDataDir), ppms[i]);
            }

            std::vector<float> data(samplesCommon::volume(mDims));
            const float scale = 2.0 / 255.0;
            const float bias = 1.0;
            long int volChl = mDims.d[2] * mDims.d[3];

            // Normalize input data
            for (int i = 0, volImg = mDims.d[1] * mDims.d[2] * mDims.d[3]; i < mBatchSize; ++i) {
                for (int c = 0; c < mDims.d[1]; ++c) {
                    for (int j = 0; j < volChl; ++j) {
                        data[i * volImg + c * volChl + j] = scale * float(ppms[i].buffer[j * mDims.d[1] + c]) - bias;
                    }
                }
            }

            std::copy_n(data.data(), mDims.d[0] * mImageSize, getFileBatch());
        }

        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    std::vector<float> mBatch;         //!< Data for the batch
    std::vector<float> mLabels;        //!< Labels for the batch
    std::vector<float> mFileBatch;     //!< List of image files
    std::vector<float> mFileLabels;    //!< List of label files
    std::string mPrefix;               //!< Batch file name prefix
    std::string mSuffix;               //!< Batch file name suffix
    nvinfer1::Dims mDims;              //!< Input dimensions
    std::string mListFile;             //!< File name of the list of image names
    std::vector<std::string> mDataDir; //!< Directories where the files can be found
};

#endif
