/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 14:06:38
 * @LastEditTime: 2019-12-10 16:55:06
 * @LastEditors: zerollzeng
 */
#include "Trt.hpp"
#include "utils.h"
//#include "spdlog/spdlog.h"
#include "Int8EntropyCalibrator.hpp"
// #include "tensorflow/graph.pb.h"

#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <memory>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
using nvinfer1::OptProfileSelector;
using nvinfer1::ILogger;
#include "NvInferPlugin.h"

extern thread_local  std::function<std::vector<float>(int) > g_calib;


Trt::Trt() {
    mNetBatchSize = 0;
	//mPluginFactory = nvonnxparser::createPluginFactory(mLogger);
	cudaStreamCreate(&cStream);
}



Trt::~Trt() {
    cudaStreamDestroy(cStream);
    
    if(mContext != nullptr) {
        mContext->destroy();
        mContext = nullptr;
    }
    if(mEngine !=nullptr) {
        mEngine->destroy();
        mEngine = nullptr;
    }
    for(size_t i=0;i<mBinding.size();i++) {
        safeCudaFree(mBinding[i]);
    }
}

bool Trt::CreateEngine(const std::string& prototxt, 
                       const std::string& caffeModel,
                       std::string& engineFile,
                       const std::vector<std::string>& outputBlobName,
                       std::function<std::vector<float>(int)> next_batch,
                       int maxBatchSize,
                       int mode) {
    mRunMode = mode;
    mBatchSize = maxBatchSize;
    //spdlog::info("prototxt: {}",prototxt);
    //spdlog::info("caffeModel: {}",caffeModel);
    //spdlog::info("engineFile: {}",engineFile);
    for(size_t i=0;i<outputBlobName.size();i++) {
        std::cout << outputBlobName[i] << " ";
    }
    std::cout << std::endl;
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngine(prototxt,caffeModel,engineFile,outputBlobName,next_batch,maxBatchSize)) {
            return false;
        }
    }
    InitEngine();
    // Notice: close profiler
    //mContext->setProfiler(mProfiler);
    return true;
}

bool Trt::CreateEngine(const std::string& onnxModel,
                       std::string& engineFile,
                       const std::vector<std::string>& customOutput,
                       int maxBatchSize,
                       int mode) {
              mBatchSize =    maxBatchSize;    
              mRunMode = mode;  
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngine(onnxModel,engineFile,customOutput,maxBatchSize)) {
            return false;
        }
    }
    InitEngine();
	return true;
}

void Trt::CreateEngine(const std::string& uffModel,
                       const std::string& engineFile,
                       const std::vector<std::string>& inputTensorNames,
                       const std::vector<std::vector<int>>& inputDims,
                       const std::vector<std::string>& outputTensorNames,
                       int maxBatchSize) {
    if(!DeserializeEngine(engineFile)) {
        if(!BuildEngine(uffModel,engineFile,inputTensorNames,inputDims, outputTensorNames,maxBatchSize)) {
            return;
        }
    }
    InitEngine();
}
void Trt::inputPointerSet_Cao(void *p) {
    tempInputBinding = mBinding[0];
    mBinding[0] = p;
}

void Trt::inputPointerRecover_Cao() {
    mBinding[0] = tempInputBinding;
}

void Trt::Forward_Cao(int bacthsize) {
    assert(bacthsize <=mBatchSize);
    mContext->enqueue(bacthsize, &mBinding[0], cStream, nullptr);
}

void Trt::Forward() {
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    mContext->execute(mBatchSize, &mBinding[0]);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
}

void Trt::ForwardAsync() {
    int nbBindings = mEngine->getNbBindings();
    for(int i=0; i< nbBindings; i++) {
        nvinfer1::Dims dims = mContext->getBindingDimensions(i);
        //if(dims.d[0] == -1)

        // if(mEngine->bindingIsInput(i)) 
        // {
        //     mBatchSize = mBindingSize[i]/(dims.d[1]*dims.d[2]*dims.d[0]*sizeof(float));
        //     dims.d[0] = mBatchSize;
        //     mContext->setBindingDimensions(i, dims);
			
        // }
        
        // if(dims.d[0] != mBatchSize)
        // {
        //     dims.d[0] = mBatchSize;
        //     exit(-1);
        // }
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        const char* name = mEngine->getBindingName(i);
        int64_t totalSize = volume(dims) * getElementSize(dtype) * mBatchSize;
        mBindingSize[i] = totalSize;
        mBindingName[i] = name;
        mBindingDims[i] = dims;
        mBindingDataType[i] = dtype;
    }
    
    mContext->enqueue(mBatchSize, &mBinding[0], cStream, nullptr);
}

void Trt::DataTransfer(std::vector<float>& data, int bindIndex, bool isHostToDevice) {
    if(isHostToDevice) {

        assert(data.size()*sizeof(float) == mBindingSize[bindIndex]);
        CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], data.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice));
    } else {
        data.resize(mBindingSize[bindIndex]/sizeof(float));
        CUDA_CHECK(cudaMemcpy(data.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost));
    }
}

void Trt::DataTransfer_input(const std::vector<float>& data, int bindIndex)
{

	//assert(data.size()*sizeof(float) == mBindingSize[bindIndex]);
    mBindingSize[bindIndex] = data.size()*sizeof(float);
	CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], data.data(), data.size()*sizeof(float), cudaMemcpyHostToDevice));

}

void Trt::DataTransfer_inputAsync(const std::vector<float>& data, int bindIndex)
{
 
    int nbBindings = mEngine->getNbBindings();

    for(int i=0; i< nbBindings; i++) { 
            
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
 

        if(mEngine->bindingIsInput(i)) 
        {
            mBatchSize = data.size()/(dims.d[1]*dims.d[2]*dims.d[0]);
            

            // if (dims.d[0] != mBatchSize)
            // {
            //     dims.d[0] = mBatchSize;
            //     if (mNetBatchSize == -1)
            //         mContext->setBindingDimensions(i, dims);
            //     nvinfer1::Dims dims2 = mEngine->getBindingDimensions(i);
            //     std::cout << dims.d[0] << " " << dims2.d[0] << std::endl;
            //     exit(-1);
            // }
            

        }
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        const char* name = mEngine->getBindingName(i);

        // if (mNetBatchSize == -1)
        // {
        //     int64_t totalSize = volume(dims)  * getElementSize(dtype);
        //     mBindingSize[i] = totalSize;
        //     mBindingName[i] = name;
        //     mBindingDims[i] = dims;
        //     mBindingDataType[i] = dtype;
        //     if(mEngine->bindingIsInput(i)) {
        //     } else {
        //     }
        // }
        // else
        {
            int64_t totalSize = volume(dims)  * getElementSize(dtype)* mBatchSize;
            mBindingSize[i] = totalSize;
            mBindingName[i] = name;
            mBindingDims[i] = dims;
            mBindingDataType[i] = dtype;

        }
    }

	 
//cudaHostRegister(data.data(), data.size()*sizeof(float),0);
	mBindingSize[bindIndex] = data.size()*sizeof(float);
     
	//assert(data.size()*sizeof(float) == mBindingSize[bindIndex]);
	CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], data.data(), data.size()*sizeof(float), cudaMemcpyHostToDevice, cStream));

}


void Trt::DataTransfer_inputAsync(float* pGpuData, int len, int bindIndex)
{
int nbBindings = mEngine->getNbBindings();

    for(int i=0; i< nbBindings; i++) { 
            
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
 
       
        if(mEngine->bindingIsInput(i)) 
        {
            
            mBatchSize = len/(dims.d[1]*dims.d[2]*dims.d[0]);
           
            

            // if (dims.d[0] != mBatchSize)
            // {
            //     dims.d[0] = mBatchSize;
            //     if (mNetBatchSize == -1)
            //         mContext->setBindingDimensions(i, dims);
            //     nvinfer1::Dims dims2 = mEngine->getBindingDimensions(i);
            //     std::cout << dims.d[0] << " " << dims2.d[0] << std::endl;
            //     exit(-1);
            // }
            

        }
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        const char* name = mEngine->getBindingName(i);

        // if (mNetBatchSize == -1)
        // {
        //     int64_t totalSize = volume(dims)  * getElementSize(dtype);
        //     mBindingSize[i] = totalSize;
        //     mBindingName[i] = name;
        //     mBindingDims[i] = dims;
        //     mBindingDataType[i] = dtype;
        //     if(mEngine->bindingIsInput(i)) {
        //     } else {
        //     }
        // }
        // else
        {
            int64_t totalSize = volume(dims)  * getElementSize(dtype)* mBatchSize;

             

            mBindingSize[i] = totalSize;
            mBindingName[i] = name;
            mBindingDims[i] = dims;
            mBindingDataType[i] = dtype;

        } 
    }


    //cudaHostRegister(data.data(), data.size()*sizeof(float),0);
	mBindingSize[bindIndex] = len*sizeof(float);
     
	assert(len*sizeof(float) == mBindingSize[bindIndex]);

    //std::cout <<len*sizeof(float)<< std::endl;
    //cudaMalloc((void **)(mBinding[bindIndex]), len*sizeof(float));
    //pGpuData = (float*)safeCudaMalloc(len*sizeof(float));
    //std::cout <<len*sizeof(float) << " "<< (void*)pGpuData << std::endl;
  
  
	CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], pGpuData, len*sizeof(float),\
         cudaMemcpyDeviceToDevice, cStream));
 }


void Trt::DataTransferAsync( std::vector<float>& data, int bindIndex, bool isHostToDevice) {
    if(isHostToDevice) {
        assert(data.size()*sizeof(float) == mBindingSize[bindIndex]);
        CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], data.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice, cStream));
    } else {
        //std::cout <<"mBindingSize[bindIndex] "<<mBindingSize[bindIndex] << std::endl;
        data.resize(mBindingSize[bindIndex]/sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(data.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost, cStream));
    }
}

void Trt::CopyFromHostToDevice(const std::vector<float>& input, int bindIndex) {
    CUDA_CHECK(cudaMemcpy(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice));
}

void Trt::CopyFromHostToDeviceAsync(const std::vector<float>& input, int bindIndex) {
    CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], input.data(), mBindingSize[bindIndex], cudaMemcpyHostToDevice, cStream));
}

void Trt::CopyFromDeviceToHost(std::vector<float>& output, int bindIndex) {
    CUDA_CHECK(cudaMemcpy(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost));
}

void Trt::CopyFromDeviceToHostAsync(std::vector<float>& output, int bindIndex) {
    CUDA_CHECK(cudaMemcpyAsync(output.data(), mBinding[bindIndex], mBindingSize[bindIndex], cudaMemcpyDeviceToHost, cStream));
}

void Trt::SetDevice(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

int Trt::GetDevice() const { 
    int* device = nullptr; //NOTE: memory leaks here
    CUDA_CHECK(cudaGetDevice(device));
    if(device != nullptr) {
        return device[0];
    } else {
        return -1;
    }
}

int Trt::GetMaxBatchSize() const{
    return mBatchSize;
}

void* Trt::GetBindingPtr(int bindIndex) const {
    return mBinding[bindIndex];
}

size_t Trt::GetBindingSize(int bindIndex) const {
    return mBindingSize[bindIndex];
}

nvinfer1::Dims Trt::GetBindingDims(int bindIndex) const {
    return mBindingDims[bindIndex];
}

nvinfer1::DataType Trt::GetBindingDataType(int bindIndex) const {
    return mBindingDataType[bindIndex];
}

void Trt::SaveEngine(const std::string& fileName) {
    if(fileName == "") {
        return;
    }
    if(mEngine != nullptr) {
        nvinfer1::IHostMemory* data = mEngine->serialize();
        std::ofstream file;
        file.open(fileName,std::ios::binary | std::ios::out);
        if(!file.is_open()) {
            return;
        }
        file.write((const char*)data->data(), data->size());
        file.close();
        data->destroy();
    } else {
    }
}

template <typename T>
bool Trt::DeserializeEngineData(const T& enginedata) {
	/*
	Q: How do I use TensorRT on multiple GPUs?
	A: Each ICudaEngine object is bound to a specific GPU when it is 
	instantiated, either by the builder or on deserialization. To select 
	the GPU, use cudaSetDevice() before calling the builder or deserializing the
	 engine. Each IExecutionContext is bound to the same GPU as the engine
	 from which it was created. When calling execute() or enqueue(),
	 ensure that the thread is associated with the correct device 
	 by calling cudaSetDevice() if necessary.
	 
	 #include <cuda.h>
	 cudaSetDevice(1);
	 cudaGetDeviceProperties
	 cudaGetDevice()
	 
	cudaGetDeviceCount(&count);if (count == 0) {
    fprintf(stderr, "There is no device.\n");
    return false;
	}

	 */
	std::cout<<"Trt::DeserializeEngineData Ln426 1"<<std::endl;
	size_t bufCount = enginedata.size();

	initLibNvInferPlugins(&mLogger, "");
    std::cout<<"Trt::DeserializeEngineData 2"<<std::endl;
	mRuntime = nvinfer1::createInferRuntime(mLogger);
    std::cout<<"Trt::DeserializeEngineData 3"<<std::endl;
	mEngine = mRuntime->deserializeCudaEngine(const_cast<char*>(enginedata.data()), bufCount, nullptr);
    std::cout<<"Trt::DeserializeEngineData 4"<<std::endl;
	assert(mEngine != nullptr);
	//mBatchSize = mEngine->getMaxBatchSize();
    std::cout<<"Trt::DeserializeEngineData 5"<<std::endl;
	mRuntime->destroy();
    std::cout<<"Trt::DeserializeEngineData 6"<<std::endl;
	return true;
    
}

bool Trt::DeserializeEngine(const std::string& engineFile) {
    std::cout<<"Trt::DeserializeEngine Ln440 1"<<std::endl;
    std::ifstream in(engineFile.c_str(), std::ifstream::binary);
    std::cout<<"Trt::DeserializeEngine 2"<<std::endl;
    if(in.is_open()) {
        std::cout<<"Trt::DeserializeEngine 3"<<std::endl;
        auto const start_pos = in.tellg();
        in.ignore(std::numeric_limits<std::streamsize>::max());
        size_t bufCount = in.gcount();
        in.seekg(start_pos);
        std::cout<<"Trt::DeserializeEngine 4"<<std::endl;
        //std::unique_ptr<char[]> engineBuf(new char[bufCount]);
		std::vector<char> engineBuf(bufCount);
        in.read(engineBuf.data(), bufCount);
        std::cout<<"Trt::DeserializeEngine 5"<<std::endl;
        return DeserializeEngineData(engineBuf);
    }
	else if (engineFile.size() > 2048)
	{
        std::cout<<"Trt::DeserializeEngine 6"<<std::endl;
		return DeserializeEngineData(engineFile);
	}
    std::cout <<engineFile<<std::endl;
    return false;
}






bool Trt::BuildEngine(const std::string& prototxt, 
                        const std::string& caffeModel,
                        std::string& engineFile,
                        const std::vector<std::string>& outputBlobName,
                        std::function<std::vector<float >  (int)> next_batch,
                        int maxBatchSize) {
		assert(engineFile.empty());
        mBatchSize = maxBatchSize;
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
        assert(builder != nullptr);
        // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 
		const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
		// auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
		
        assert(network != nullptr);
        nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
        /*if(mPluginFactory != nullptr) {
            parser->setPluginFactoryV2(mPluginFactory);
        }*/
        // Notice: change here to costom data type
        nvinfer1::DataType type = mRunMode==1 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
        /*
        const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(prototxt.c_str(),caffeModel.c_str(),
                                                                                *network,type);
                                                                                */
        
        const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = 
        parser->parseBuffers(prototxt.c_str(),prototxt.size(), caffeModel.c_str(),caffeModel.size(),
                                                                                *network,type);
        /*
        parseBuffers (const char *deployBuffer, std::size_t deployLength,
         const char *modelBuffer, std::size_t modelLength, 
        nvinfer1::INetworkDefinition &network, nvinfer1::DataType weightType)=0
        */
        for(auto& s : outputBlobName) {
            network->markOutput(*blobNameToTensor->find(s.c_str()));
        }

        for(int i = 0; i < network->getNbInputs(); i++) {
            nvinfer1::Dims dims = network->getInput(i)->getDimensions();
            dims.d[0] = -1;
            network->getInput(i)->setDimensions(dims);
            /*for(int j = 0; j < dims.nbDims; j++) {
                std::cout << dims.d[j] << "x"; 
            }
            std::cout << "\b "  << std::endl;*/
        }
        for(int i = 0; i < network->getNbOutputs(); i++) {
            nvinfer1::Dims dims = network->getOutput(i)->getDimensions();
            /*for(int j = 0; j < dims.nbDims; j++) {
                std::cout << dims.d[j] << "x"; 
            }
            std::cout << "\b " << std::endl;*/
        }
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

        Int8EntropyCalibrator* calibrator = nullptr;
        if (mRunMode == 2)
        {
            if (!builder->platformHasFastInt8()) {
            }
            if (next_batch){
                auto endPos= prototxt.find_last_of(".");
                auto beginPos= prototxt.find_last_of('/') + 1;
                if(prototxt.find("/") == std::string::npos) {
                    beginPos = 0;
                }
                std::string calibratorName = prototxt.substr(beginPos,endPos - beginPos);
                calibrator = new Int8EntropyCalibrator(maxBatchSize,g_calib,calibratorName,false);
            }
            // enum class BuilderFlag : int
            // {
            //     kFP16 = 0,         //!< Enable FP16 layer selection.
            //     kINT8 = 1,         //!< Enable Int8 layer selection.
            //     kDEBUG = 2,        //!< Enable debugging of layers via synchronizing after every layer.
            //     kGPU_FALLBACK = 3, //!< Enable layers marked to execute on GPU if layer cannot execute on DLA.
            //     kSTRICT_TYPES = 4, //!< Enables strict type constraints.
            //     kREFIT = 5,        //!< Enable building a refittable engine.
            // };
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            config->setInt8Calibrator(calibrator);
        }
        
        if (mRunMode == 1)
        {
            if (!builder->platformHasFastFp16()) {
            }
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        builder->setMaxBatchSize(mBatchSize);
        // set the maximum GPU temporary memory which the engine can use at execution time.
        config->setMaxWorkspaceSize(40 << 20);
        
		auto profile = builder->createOptimizationProfile();
		//input_name = network->getInput(0).name;
        auto dim= network->getInput(0)->getDimensions();
        auto dim_min = dim;
        dim_min.d[0] = 1 ;
        auto dim_opt = dim;
        dim_opt.d[0] = mBatchSize/2 >0 ? mBatchSize/2 >0:1;
        auto dim_max = dim;
        dim_max.d[0] = mBatchSize ;
        //input_name = network->getInput(0).name;
        profile->setDimensions("data", OptProfileSelector::kMIN,  dim_min);
        profile->setDimensions("data", OptProfileSelector::kOPT, dim_opt);
        profile->setDimensions("data", OptProfileSelector::kMAX, dim_max);

		config->addOptimizationProfile(profile);



        mEngine = builder -> buildEngineWithConfig(*network, *config);
        assert(mEngine != nullptr);
        //spdlog::info("serialize engine to {}", engineFile);
        ///SaveEngine(engineFile);
        
		
		if(engineFile.empty())
		{
			nvinfer1::IHostMemory* data = mEngine->serialize();
			//file.write((const char*)data->data(), data->size());
			std::string tmpstr((const char*)data->data(), data->size());
			engineFile = tmpstr;
			data->destroy();
		}
	
        builder->destroy();
        config->destroy();
        network->destroy();
        parser->destroy();
        if(calibrator){
            delete calibrator;
            calibrator = nullptr;
        }
        return true;
}

bool Trt::BuildEngine(const std::string& onnxModel,
                      std::string& engineFile,
                      const std::vector<std::string>& customOutput,
                      int maxBatchSize) {   
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 1 ^_^"<<std::endl;
    mBatchSize = maxBatchSize;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
    assert(builder != nullptr);
    // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 
    //nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 2"<<std::endl;
	
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 2.1"<<std::endl;
    assert(network != nullptr);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, mLogger);
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 2.2"<<std::endl;
	std::ifstream fin(onnxModel);
	bool bIsfile = true;
	
	if (!fin)
	{
	   std::cout << "warning: can not open onnx file, parse as data" << std::endl;
	   bIsfile=false;
	}
	fin.close(); 
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 3"<<std::endl;
	
    if (false) 
    {
        parser->parseFromFile(
            "/home/zhangsy/tensorrt7/tiny_tensorrt_vtn/model/person-reidentification-retail-0200_dynamic.onnx", 
            static_cast<int>(ILogger::Severity::kWARNING));
    }
    else
    { 
        if (!bIsfile){
            std::cout<<"src/tensorrtcppwrapper/trt.cpp 3.1"<<std::endl;
            if (!parser->parse(onnxModel.data(),onnxModel.size()))
            {
                std::cout<<"src/tensorrtcppwrapper/trt.cpp 3.2"<<std::endl;
                //spdlog::error("error: could not parse onnx engine. bIsfile: {}", bIsfile);
                return false;
            }
        }
        else if(!parser->parseFromFile(onnxModel.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
            std::cout<<"src/tensorrtcppwrapper/trt.cpp 3.3"<<std::endl;
            //spdlog::error("error: could not parse onnx engine");
            return false;
        }
        std::cout<<"src/tensorrtcppwrapper/trt.cpp 3.4"<<std::endl;
    }
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 4"<<std::endl;

//    auto* origin_output = network->getOutput(0);
//    network->unmarkOutput(*origin_output);
//    uint32_t axis =  origin_output->getDimensions().nbDims - 1;
//    uint32_t axis_mask = 1 << axis;
//	auto* topKlayer = network->addTopK(*origin_output,
//	        nvinfer1::TopKOperation::kMAX,1,axis_mask);
//    network->markOutput(*(topKlayer->getOutput(0)));
//    network->markOutput(*(topKlayer->getOutput(1)));


    if(customOutput.size() > 0) {
        for(int i=0;i<network->getNbOutputs();i++) {
            nvinfer1::ITensor* origin_output = network->getOutput(i);
            network->unmarkOutput(*origin_output);
        }
        for(int i=0;i<network->getNbLayers();i++) {
            nvinfer1::ILayer* custom_output = network->getLayer(i);
            nvinfer1::ITensor* output_tensor = custom_output->getOutput(0);
            for(size_t j=0; j<customOutput.size();j++) {
                std::string layer_name(output_tensor->getName());
                if(layer_name == customOutput[j]) {
                    network->markOutput(*output_tensor);
                    break;
                }
            }
        }    
    }
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 5 mRunMode="<<mRunMode<<";"<<std::endl;
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

        Int8EntropyCalibrator* calibrator = nullptr;
    if (mRunMode == 2)
    {
            if (!builder->platformHasFastInt8()) {
                //spdlog::warn("Warning: current platform doesn't support int8 inference");
            }
            if (!g_calib)
            {
                assert(g_calib);
                return false;
            }
		
                std::string calibratorName = "calib_name_onnx";
                std::cout << "create calibrator,Named:" << calibratorName << std::endl;
                assert(maxBatchSize == 1);
                calibrator = new Int8EntropyCalibrator(maxBatchSize,g_calib,calibratorName,false);
	    config->setFlag(nvinfer1::BuilderFlag::kINT8); 
         config->setInt8Calibrator(calibrator);	
    }
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 6"<<std::endl;

    builder->setMaxBatchSize(mBatchSize);
    config->setMaxWorkspaceSize(1*mBatchSize << 20);

	/*if (network == NULL) {
        std::cout<<"network is NULL!!!!!!!!!"<<std::endl;
    }
    if (network->getInput(0) == NULL) {
        std::cout<<"network->getInput(0) is NULL"<<std::endl;
    }
    std::cout<<"Trt::buildEngine Ln727"<<std::endl;*/
	mNetBatchSize = network->getInput(0)->getDimensions().d[0];
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 7"<<std::endl;

    if (mRunMode == 0 && network->getInput(0)->getDimensions().d[0] ==-1)
    {
        auto profile = builder->createOptimizationProfile();
        auto dim= network->getInput(0)->getDimensions();
        auto dim_min = dim;
        dim_min.d[0] = 1 ;
        auto dim_opt = dim;
        dim_opt.d[0] = mBatchSize/2 >0 ? mBatchSize/2:1 ;
        auto dim_max = dim;
        dim_max.d[0] = mBatchSize ;
        //input_name = network->getInput(0).name;
        profile->setDimensions("data", OptProfileSelector::kMIN,  dim_min);
        profile->setDimensions("data", OptProfileSelector::kOPT, dim_opt);
        profile->setDimensions("data", OptProfileSelector::kMAX, dim_max);

        config->addOptimizationProfile(profile);
    }
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 8"<<std::endl;

	
		
    mEngine = builder -> buildEngineWithConfig(*network, *config);
    assert(mEngine != nullptr);

	if(engineFile.empty())
	{
		nvinfer1::IHostMemory* data = mEngine->serialize();
		//file.write((const char*)data->data(), data->size());
		std::string tmpstr((const char*)data->data(), data->size());
		engineFile = tmpstr;
        data->destroy();
	}
    std::cout<<"src/tensorrtcppwrapper/trt.cpp 9"<<std::endl;
	
    builder->destroy();
    network->destroy();
    parser->destroy();
	if(calibrator){
            delete calibrator;
            calibrator = nullptr;
        }
    return true;
}

bool Trt::BuildEngine(const std::string& uffModel,
                      const std::string& engineFile,
                      const std::vector<std::string>& inputTensorNames,
                      const std::vector<std::vector<int>>& inputDims,
                      const std::vector<std::string>& outputTensorNames,
                      int maxBatchSize) {
    mBatchSize = maxBatchSize;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mLogger);
    assert(builder != nullptr);
    // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH 
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    assert(network != nullptr);
    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
    assert(parser != nullptr);
    assert(inputTensorNames.size() == inputDims.size());
    //parse input
    for(size_t i=0;i<inputTensorNames.size();i++) {
        nvinfer1::Dims dim;
        dim.nbDims = inputDims[i].size();
        for(int j=0;j<dim.nbDims;j++) {
            dim.d[j] = inputDims[i][j];
        }
        parser->registerInput(inputTensorNames[i].c_str(), dim, nvuffparser::UffInputOrder::kNCHW);
    }
    //parse output
    for(size_t i=0;i<outputTensorNames.size();i++) {
        parser->registerOutput(outputTensorNames[i].c_str());
    }
    if(!parser->parse(uffModel.c_str(), *network, nvinfer1::DataType::kFLOAT)) {
    }
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(10 << 20);
    builder->setMaxBatchSize(mBatchSize);
    mEngine = builder -> buildEngineWithConfig(*network, *config);
    assert(mEngine != nullptr);
    SaveEngine(engineFile);

    builder->destroy();
    network->destroy();
    parser->destroy();
    return true;
}


void Trt::InitEngine() {
    //spdlog::info("init engine...");
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    //spdlog::info("malloc device memory");
    int nbBindings = mEngine->getNbBindings();
    //std::cout << "nbBingdings: " << nbBindings << std::endl;
    mBinding.resize(nbBindings);
    mBindingSize.resize(nbBindings);
    mBindingName.resize(nbBindings);
    mBindingDims.resize(nbBindings);
    mBindingDataType.resize(nbBindings);
    for(int i=0; i< nbBindings; i++) { 
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
        //if(dims.d[0] == -1)
        // if(dims.d[0] == -1)
        // {
        //     //spdlog::info("change batch from {} to {}", dims.d[0], mBatchSize);
        //     dims.d[0] = mBatchSize;
        // }
        if(mEngine->bindingIsInput(i)) 
        {
            mContext->setBindingDimensions(i, dims);
        }
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        const char* name = mEngine->getBindingName(i);
        // assert(mBatchSize == 32);
        // exit(0);
        int64_t totalSize = volume(dims)  * getElementSize(dtype)*mBatchSize;
        mBindingSize[i] = totalSize;
        mBindingName[i] = name;
        mBindingDims[i] = dims;
        mBindingDataType[i] = dtype;
        // if(mEngine->bindingIsInput(i)) {
        //     spdlog::info("input: ");
        // } else {
        //     spdlog::info("output: ");
        // }
        // spdlog::info("binding bindIndex: {}, name: {}, size in byte: {}",i,name,totalSize);
        // spdlog::info("binding dims with {} dimemsion",dims.nbDims);
        // for(int j=0;j<dims.nbDims;j++) {
        //     std::cout << dims.d[j] << " x ";
        // }
        // std::cout << totalSize<<""<<"totalSize\b\b  "<< std::endl;
        mBinding[i] = safeCudaMalloc(totalSize);
        if(mEngine->bindingIsInput(i)) {
            mInputSize++;
        }
    }
}
