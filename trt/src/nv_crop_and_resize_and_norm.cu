#include <cuda.h>
#include <cuda_runtime.h>
#include <nvCropAndResizeNovio.h>
#include <iostream>



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
    
    
    CUDA_CHECK(cudaMallocManaged(&deviceMem, memSize));
cudaDeviceSynchronize();
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
        float src_idx = scale * (dst_index + 0.5) - 0.5 + crop;
        // See Note[Follow Opencv resize logic]
        return src_idx;
    }
}
__device__ __forceinline__ unsigned char get_val(unsigned char  *data,int c, int y,int x,int W,int detx1,int detx2,int dety1,int dety2, unsigned char pad) {
    if (x < detx1 || x >= detx2 || y <dety1 || y >= dety2 ){
        return pad;
    }else{
#if __CUDA_ARCH__ >= 350
    return __ldg(&data[(y*W+x)*3+c]);
#else
    return data[(y*W+x)*3+c];
#endif
        //return __ldg(&data[(y*W+x)*3+c]);
    }
}

void freePointer(void *p){
    //CUDA_CHECK(cudaFree(p));
    //p= nullptr;
    cudaDeviceSynchronize();
    cudaFree(p);
    p = nullptr;
}


__forceinline__ __global__ void nvCropAndResizeAndNormKernel(unsigned char * Src, float * cropImage,ITS_Vehicle_Result_Detect* det,
            int srcWidth,int srcHeight,int outW,int outH,float* means,float* stds)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int detIndex = blockIdx.z;
    int detx1 = det[0].iLeft[detIndex],dety1 = det[0].iTop[detIndex];
    int detx2 = det[0].iRight[detIndex] ,dety2 = det[0].iBottom[detIndex];
    int cropW = detx2-detx1,cropH = dety2-dety1;
    float scale = max((float)cropW/outW,(float)cropH/outH);
    int shift_x = (outW - cropW/scale) / 2;
    int shift_y = (outH  - cropH/scale) / 2;
    if (x >= outW || y >= outH ||cropW <=0 ||cropH <=0)
        return;
    float x1r = area_pixel_compute_source_index(
            scale, x - shift_x,detx1,false);
    int x1 = x1r;
    int x1p = (x1 < srcWidth - 1) ? 1 : 0;
    float x1lambda = x1r - x1;
    float x0lambda = 1. - x1lambda;
    float y1r = area_pixel_compute_source_index(
            scale, y - shift_y,dety1,false);
    int y1 = y1r;
    int y1p = (y1 < srcHeight - 1) ? 1 : 0;
    float y1lambda = y1r - y1;
    float y0lambda = 1. - y1lambda;
    float *dst ;
    float val = 0 ;

#pragma unroll
    for (int channel=0; channel < 3; channel++){
        // chw
        dst = cropImage + ((detIndex*3 + channel)*outH + y)*outW + x ;
        val = x0lambda*y0lambda* static_cast<float >(get_val(Src,channel,y1,x1,srcWidth,detx1,detx2,dety1,dety2,127)) +
          x0lambda*y1lambda* static_cast<float >(get_val(Src,channel,y1+y1p,x1,srcWidth,detx1,detx2,dety1,dety2,127)) +
          x1lambda*y0lambda* static_cast<float >(get_val(Src,channel,y1,x1+x1p,srcWidth,detx1,detx2,dety1,dety2,127)) +
          x1lambda*y1lambda* static_cast<float >(get_val(Src,channel,y1+y1p,x1+x1p,srcWidth,detx1,detx2,dety1,dety2,127)) ;
        *dst = (val /255.0f-means[channel])/stds[channel];

    }
}

void nvCropAndResizeAndNormLaunch(float* cropImages,
                         const std::vector<unsigned char *> &cudaSrc,
                         ITS_Vehicle_Result_Detect *tempCudaDet,
std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                         const std::vector<int> & srcWidth,const std::vector<int> & srcHeight,
                         int batchSize,int cropW,int cropH,float means[3],float stds[3]){


    float* meansCuda = (float*)safeCudaMalloc(3 * sizeof(float));
    float* stdsCuda = (float*)safeCudaMalloc(3 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(meansCuda,means, 3*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(stdsCuda,stds, 3*sizeof(float),cudaMemcpyHostToDevice));
    int totalCars = 0,carNum;
    int perCarPixels = cropW*cropH*3;

    for(int i=0;i<cpuDet.size();++i){
        carNum = cpuDet[i].CarNum;
        if(carNum > 0){
             dim3 block(32, 32, 1);
             dim3 grid((cropW + block.x - 1) / block.x,
                      (cropH + block.y - 1) / block.y, carNum);
             if(totalCars + carNum > batchSize){
                std::cerr<<"batchSize "<<batchSize<<" not enough"<<std::endl;
                break;
             }
             nvCropAndResizeAndNormKernel<<<grid,block,0,0>>>(cudaSrc[i],cropImages+(totalCars*perCarPixels),tempCudaDet+i,
                    srcWidth[i],srcHeight[i],cropW,cropH,meansCuda,stdsCuda);
             totalCars += carNum;
        }
    }

    freePointer(meansCuda);
    freePointer(stdsCuda);
}


ITS_Vehicle_Result_Detect* initTempCudaDet(int maxDetNum){
    ITS_Vehicle_Result_Detect* cudadet= (ITS_Vehicle_Result_Detect*)safeCudaMalloc(maxDetNum * sizeof(ITS_Vehicle_Result_Detect));
    CUDA_CHECK(cudaMemset(cudadet,0,maxDetNum * sizeof(ITS_Vehicle_Result_Detect)));
    return cudadet;
}