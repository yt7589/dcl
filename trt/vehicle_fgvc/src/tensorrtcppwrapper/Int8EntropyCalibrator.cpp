/*
 * @Description: int8 entrophy calibrator 2
 * @Author: zengren
 * @Date: 2019-08-21 16:52:06
 * @LastEditTime: 2019-08-22 17:04:49
 * @LastEditors: Please set LastEditors
 */
#include "Int8EntropyCalibrator.hpp"
#include <fstream>
#include <iterator>
#include <cassert>
#include <string.h>
#include <algorithm>


Int8EntropyCalibrator::Int8EntropyCalibrator(int BatchSize,  std::function<std::vector<float >  (int)> next_batch,
                                        const std::string& CalibDataName /*= ""*/,bool readCache /*= true*/)
    : mCalibDataName(CalibDataName),mBatchSize(BatchSize),mReadCache(readCache)
{     
    mNextBatch = next_batch;

    mInputCount =  BatchSize * 1920*1080*3;
    mCurBatchData = nullptr;
    mCurBatchIdx = 0;
    CUDA_CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
}


Int8EntropyCalibrator::~Int8EntropyCalibrator()
{
    CUDA_CHECK(cudaFree(mDeviceInput));
    if(mCurBatchData)
        delete[] mCurBatchData;
}


bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    std::cout << "name: " << names[0] << " nbBindings: " << nbBindings << std::endl;
    
    auto data = mNextBatch(mBatchSize);
    std::cout << "!! mInputCount "<<  data.size()<< std::endl;
    if (data.empty())
	{
	return false;
	}
	mInputCount =  data.size();
    
    float* ptr = data.data();
    size_t imgSize = mInputCount / mBatchSize;

    CUDA_CHECK(cudaMemcpy(mDeviceInput, ptr, mInputCount * sizeof(float), cudaMemcpyHostToDevice));
    //std::cout << "input name " << names[0] << std::endl;
    bindings[0] = mDeviceInput;

    std::cout << "load batch " << mCurBatchIdx << " to " << mCurBatchIdx + mBatchSize - 1 << std::endl;        
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length)
{
    mCalibrationCache.clear();
    std::ifstream input(mCalibDataName+".calib", std::ios::binary);
    input >> std::noskipws;
    if (mReadCache && input.good())
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length)
{
    std::ofstream output(mCalibDataName+".calib", std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

