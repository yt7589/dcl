//
// Created by novio on 20-2-12.
//

#include <chrono>
#include <cuda_runtime.h>
#include <logger.h>
#include "AllModuleInclude.h"
#include "CarFeature.h"
#include "NvInferRuntimeCommon.h"
#include <vector>
#include <fstream>
#include <dirent.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <deque>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#define NUM_THREADS  1
#define SHOW_RESULT
#define NO_Detect_CarTYPE_WITH_CPU

vector< string> split(string str, string pattern)
{
    vector<string> ret;
    if (pattern.empty()) return ret;
    size_t start = 0, index = str.find_first_of(pattern, 0);
    while (index != str.npos)
    {
        if (start != index)
            ret.push_back(str.substr(start, index - start));
        start = index + 1;
        index = str.find_first_of(pattern, start);
    }
    if (!str.substr(start).empty())
        ret.push_back(str.substr(start));
    return ret;
}

std::vector<std::string> for_each_file(const std::string& dir_name){
    std::vector<std::string> v;
    auto dir =opendir(dir_name.data());
    struct dirent *ent;
    if(dir){
          while ((ent = readdir (dir)) != NULL) {
               auto p = std::string(dir_name).append({ '/' }).append(ent->d_name);
               if ( 0== strcmp (ent->d_name, "..") || 0 == strcmp (ent->d_name, ".")){
                   continue;
               }
               else{
                   v.emplace_back(p);
               }
          }
    }
    closedir(dir);
    return v;

}

void *mythread(void *threadid) {
    int tid = *((int *) threadid);
    int batchSize = 1;
    int gpus{0};
    cudaGetDeviceCount(&gpus);
    void *handler = VehicleFeatureInstance("/hd10t/yantao/dcl/trt/onnx2trt_int8/models/", tid % gpus, 8);
    int iDebug = 1;
    if (1 == iDebug) 
    {
        return NULL;
    }
    std::cout << "init finish" << std::endl;
    
    std::vector<std::string> all_img_list = for_each_file(
            "/home/up/wang/feature_test/raw_data/gallery_concat/");
    ifstream in("/home/up/wang/feature_test/raw_data/gallery_0423_new.txt");
    std::string save_path = "/home/up/wang/feature_test/result/gallery/";
    uint8_t *ptr;
    cudaMalloc(&ptr, 384 * 384 * 3 * sizeof(uint8_t));
    for (string s; getline(in, s);)
    {
        string pattern = " ";
        string pattern1 = ",";
        vector<string> line_result = split(s, pattern);
        vector<string> coor = split(line_result[2], pattern1);
        int left = atoi(coor[0].c_str());
        int top = atoi(coor[1].c_str());
        int right = atoi(coor[2].c_str());
        int bottom = atoi(coor[3].c_str());

    //}
    //for(int ii = 0; ii < all_img_list.size(); ++ii){
    //cv::Mat testImage = cv::imread(all_img_list[ii]);
    //std::cout << all_img_list[ii] << std::endl;
        string base_path = "/home/up/wang/feature_test/";
        cv::Mat testImage = cv::imread(base_path + line_result[0]);
        cv::resize(testImage, testImage, cv::Size(384, 384));
        int total_times = 1;
        auto start = std::chrono::system_clock::now();
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::vector<std::vector<CAR_FEATURE_RESULT>> results;
        std::vector<unsigned char *> testImagesGpu{};

        cudaMemcpy(ptr, testImage.data, 384 * 384 * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
        std::vector<int> heights(batchSize, 384), widths(batchSize, 384);
        for (int i = 0; i < batchSize; ++i) {
            testImagesGpu.emplace_back(ptr);
        }

        start = std::chrono::system_clock::now();
        std::vector<ITS_Vehicle_Result_Detect> dets;
        std::vector<ITS_Vehicle_Result_Detect> pdets;
        for (int i = 0; i < batchSize; ++i) {
            ITS_Vehicle_Result_Detect tmp;
            tmp.CarNum = 1;
            for (int j= 0; j < tmp.CarNum ; j++){
                tmp.fConfdence[j] = 1.0f;
                tmp.iLeft[j] = 0;
                tmp.iTop[j] = 0;
                tmp.iRight[j] = 384;
                tmp.iBottom[j] = 384;
            }
            dets.emplace_back(tmp);

            ITS_Vehicle_Result_Detect tmp2;
            tmp2.CarNum = 1;
            for (int j= 0; j < tmp2.CarNum ; j++){
                tmp2.fConfdence[0] = 1.0f;
                tmp2.iLeft[j] = left;
                tmp2.iTop[j] = top;
                tmp2.iRight[j] = right;
                tmp2.iBottom[j] = bottom;
            }
            pdets.emplace_back(tmp2);
        }
    //for (int i = 0; i < all_img_list.size(); ++i) {
        //results = GetMaskedCarFeatureGPU(handler, testImagesGpu, heights, widths, dets, pdets);
        results = GetCarFeatureGPU(handler, testImagesGpu, heights, widths, dets);
#ifdef SHOW_RESULT
        for (int j = 0; j < results.size(); ++j) {
            for (auto &mResult : results[j]){
                std::cout << "first ten feature:" << std::endl;
                auto parsedFeature = parseFEATUREVector(mResult);
                ofstream ouF;
                //std::string image_name = all_img_list[ii].substr(all_img_list[ii].find_last_of("/") + 1);
                std::string image_name = line_result[0].substr(line_result[0].find_last_of("/") + 1);
                ouF.open(save_path + image_name + ".bin", std::ofstream::binary);
                ouF.write(reinterpret_cast<const char*>(parsedFeature.data()), sizeof(float)*256);
                ouF.close();
                for (int k = 0; k < 10; ++k) {
                    std::cout << parsedFeature[k] << "\t";
                }
                std::cout << std::endl;
            }
        }
    #endif
        //}
        end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "gpu image pointer 平均每次请求花费了"
                  << double(duration.count()) * std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den / total_times
                  << "秒" << std::endl;
        std::cout << "finish" << std::endl;

    }
    cudaFree(ptr);
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
