//
// Created by novio on 20-2-12.
//
//#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/build/lib/libopencv_core.so.3.4"
#include <chrono>
#include "/usr/local/cuda-10.1/targets/x86_64-linux/include/cuda_runtime.h"
#include "logger.h"
#include "fgvc.h"
#include "/home/zjkj/working_zjw/onnx--prog/TensorRT-6.0.1.5/include/NvInferRuntimeCommon.h"
#include <vector>

#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/modules/core/include/opencv2/core.hpp"
#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/modules/imgcodecs/include/opencv2/imgcodecs.hpp"
#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/modules/imgproc/include/opencv2/imgproc.hpp"
//#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/build/lib/libopencv_core.so.3.4"
//#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/build/lib/libopencv_core.so.3.4"
#define NUM_THREADS   1
//#define SHOW_RESULT
#define NO_Detect_CarTYPE_WITH_CPU
#define IMG_H 448
#define IMG_W 448


void *mythread(void *threadid) {
    int tid = *((int *) threadid);
    int batchSize = 8;
    int gpus{0};
    cudaGetDeviceCount(&gpus);
    void *handler = VehicleFeatureInstance("./", tid % gpus, 32);
    std::cout << "init finish" << std::endl;
    cv::Mat testImage = cv::imread("./test_image.jpg");
    cv::resize(testImage, testImage, cv::Size(IMG_W, IMG_H));
    int total_times = 100;
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::vector<std::vector<CAR_FEATURE_RESULT>> results;
    std::vector<unsigned char *> testImagesGpu{};
    uint8_t *ptr;
    cudaMalloc(&ptr, IMG_W * IMG_H * 3 * sizeof(uint8_t));
    cudaMemcpy(ptr, testImage.data, IMG_W * IMG_H * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    std::vector<int> heights(batchSize, IMG_H), widths(batchSize, IMG_W);
    for (int i = 0; i < batchSize; ++i) {
        testImagesGpu.emplace_back(ptr);
    }

    start = std::chrono::system_clock::now();
    std::vector<ITS_Vehicle_Result_Detect> dets;

    for (int i = 0; i < batchSize; ++i) {
        ITS_Vehicle_Result_Detect tmp;
        tmp.CarNum = 1;
        tmp.fConfdence[0] = 1.0f;
        tmp.iLeft[0] = 0;
        tmp.iTop[0] = 0;
        tmp.iRight[0] = IMG_W;
        tmp.iBottom[0] = IMG_H;
        dets.emplace_back(tmp);
    }
    for (int i = 0; i < total_times; ++i) {
        results = GetCarFeatureGPU(handler, testImagesGpu, heights, widths, dets);
#ifdef SHOW_RESULT
        for (int j = 0; j < results.size(); ++j) {
            for (auto &mResult : results[j]){
                std::cout << "first ten feature:" << std::endl;
                auto parsedFeature = parseFEATUREVector(mResult);
                for (int k = 0; k < 10; ++k) {
                    std::cout << parsedFeature[k] << "\t";
                }
                std::cout << std::endl;
            }

        }
#endif
    }
    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "gpu image pointer 平均每次请求花费了"
              << double(duration.count()) * std::chrono::microseconds::period::num /
                 std::chrono::microseconds::period::den / total_times
              << "秒" << std::endl;
    std::cout << "finish" << std::endl;
    ReleaseSDKFeature(handler);

}


int main() {
    setReportableSeverity(Severity::kERROR);
    pthread_t threads[NUM_THREADS];
    int indexes[NUM_THREADS];// 用数组来保存i的值
    for (int i = 0; i < NUM_THREADS; i++) {
        indexes[i] = i; //先保存i的值
// 传入的时候必须强制转换为void* 类型，即无类型指针
        int rc = pthread_create(&threads[i], NULL, mythread, (void *) &(indexes[i]));
        if (rc) {
            exit(-1);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
//cout << "right:创建线程," << i << endl;
    }
    printf("end\n");
    //ITS_VehicleRecRelease(pInstance0);
    return 0;
}
