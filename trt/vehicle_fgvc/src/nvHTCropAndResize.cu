#include <cuda.h>
#include <cuda_runtime.h>
#include <nvHTCropAndResize.h>
#include <iostream>

typedef struct {
    float num1;
    float num2;
    float num3;
} number3;

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

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

__device__ __forceinline__  float area_pixel_compute_source_index(
        float scale,
        int dst_index,
        float crop,
        bool align_corners) {
    if (align_corners) {
        return scale * dst_index + crop;
    } else {
        float src_idx = scale * (dst_index + static_cast<float >(0.5)) -
                        static_cast<float>(0.5) + crop;
        // See Note[Follow Opencv resize logic]
        return (src_idx < static_cast<float>(0))
               ? static_cast<float>(0)
               : src_idx;
    }
}

__global__ void nvHTCropAndReizeKernel(unsigned char * Src, float * cropImage,ITS_Vehicle_Result_Detect* det,
            int srcWidth,int srcHeight,int outW,int outH,number3 mean,number3 std)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int detIndex = blockIdx.z;
    int shiftX = det[0].iLeft[detIndex],shiftY = det[0].iTop[detIndex];
    int cropW = det[0].iRight[detIndex] - shiftX,cropH = det[0].iBottom[detIndex] - shiftY;
    float scaleX = (float)cropW/outW,scaleY = (float)cropH/outH;
    if (x >= outW || y >= outH ||cropW <=0 ||cropH <=0)
        return;
    float x1r = area_pixel_compute_source_index(
            scaleX, x, shiftX,false);
    int x1 = x1r;
    int x1p = (x1 < srcWidth - 1) ? 1 : 0;
    float x1lambda = x1r - x1;
    float x0lambda = 1. - x1lambda;
    float y1r = area_pixel_compute_source_index(
            scaleY, y, shiftY,false);
    int y1 = y1r;
    int y1p = (y1 < srcHeight - 1) ? 1 : 0;
    float y1lambda = y1r - y1;
    float y0lambda = 1. - y1lambda;
    float *dst ;
    float val = 0 ;
#pragma unroll
    for (int channel=0; channel < 3; channel++){
        dst = cropImage + ((detIndex*3 + channel)*outH + y )*outW + x ;
        val = x0lambda*y0lambda* static_cast<float >(__ldg(&Src[(y1*srcWidth + x1)*3 + channel])) +
              x0lambda*y1lambda* static_cast<float >(__ldg(&Src[((y1+y1p)*srcWidth + x1)*3 + channel])) +
              x1lambda*y0lambda* static_cast<float >(__ldg(&Src[(y1*srcWidth + x1+x1p)*3 + channel])) +
              x1lambda*y1lambda* static_cast<float >(__ldg(&Src[((y1+y1p)*srcWidth + x1+x1p)*3 + channel])) ;
        *dst = ( val/255. - ((float*)&mean.num1)[channel] ) / ((float*)&std.num1)[channel] ;
    }
}

void nvHTCropAndReizeLaunch(std::vector<float*> &cropImages,
                         std::vector<unsigned char *> &cudaSrc,
                         std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                         ITS_Vehicle_Result_Detect *tempCudaDet,
                         std::vector<int> & srcWidth,std::vector<int> & srcHeight,
                         std::vector<float > & mean,std::vector<float > & std,
                         int batchSize,int cropW,int cropH){
    number3 cpuMean = {mean[0],mean[1],mean[2]};
    number3 cpuStd = {std[0],std[1],std[2]};
    CUDA_CHECK(cudaMemcpy(tempCudaDet,cpuDet.data(), batchSize*sizeof(ITS_Vehicle_Result_Detect),cudaMemcpyHostToDevice));
    for(int i=0,carNum = 0;i<batchSize;++i){
        carNum = cpuDet[i].CarNum;
        if(carNum > 0){
            dim3 block(32, 32, 1);
            dim3 grid((cropW + block.x - 1) / block.x,
                      (cropH + block.y - 1) / block.y, carNum);
            nvHTCropAndReizeKernel<<<grid,block,0,0>>>(cudaSrc[i],cropImages[i],tempCudaDet+i,
                    srcWidth[i],srcHeight[i],cropW,cropH,cpuMean,cpuStd);
        }
    }
}

std::vector<float *> initCropAndResizeImages(int cardNum,int batchSize,int maxDetNum, int maxOutWidth,int maxOutHeight){
    cudaSetDevice(cardNum);
    std::vector<float *> cromimages;
    for(int i = 0;i< batchSize;++i){
        float* cudamem = (float*)safeCudaMalloc(maxDetNum*maxOutWidth*maxOutHeight*3* sizeof(float));
        cromimages.push_back(cudamem);
    }
    return cromimages;
}

ITS_Vehicle_Result_Detect* initTempCudaDet(int cardNum,int maxDetNum){
    cudaSetDevice(cardNum);
    ITS_Vehicle_Result_Detect* cudadet= (ITS_Vehicle_Result_Detect*)safeCudaMalloc(maxDetNum * sizeof(ITS_Vehicle_Result_Detect));
    return cudadet;
}