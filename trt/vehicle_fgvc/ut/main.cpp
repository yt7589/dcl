//#include "vehicle.h"
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
//common
#include "file_util.hpp"
//#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include "aes_api.hpp"
#include "vehicle_fgvc.h"
#include <cuda_runtime.h>
#include <iostream>
#define NUM_THREADS 1

const int IMG_W = 224;
const int IMG_H = 224;
int big_batchsize = 8;
int small_batchsize = 8;


const bool GPU_INPUT = true;
const bool GPU_DETECT_INPUT = true;
void  initDet(ITS_Vehicle_Result_Detect &det,const int detNum){
    det.CarNum = detNum;
    for(int i=0; i< detNum; ++i) {
        if (i%2 == 0)
        {
            det.iLeft[i] =0;
            det.iTop[i] = 0;
            det.iRight[i] = IMG_W; // 224
            det.iBottom[i] = IMG_H; //224
        }
        else
        {
            det.iLeft[i] = IMG_W; //224;
            det.iTop[i] = IMG_H; //224;
            det.iRight[i] = 2*IMG_W; //448;
            det.iBottom[i] = 2*IMG_H; //448;
        }
        
    }
}


std::vector<float> _preProcess(const std::vector<cv::Mat> &images)
{
    auto mean = 0.485 * 255;
    auto xFactor = 1 / (0.225 * 255);

    float meanb = mean;
    float meang = mean;
    float meanr = mean;
    float xfactorb = xFactor;
    float xfactorg = xFactor;
    float xfactorr = xFactor;

    auto batch = images.size(); //_config.inputShape[0];
    //assert(batch <= conf.inputShape[0]);
    auto depth =3;
    auto height = IMG_H; //224;
    auto width = IMG_W; //224;
    // 批处理识别
    std::vector<cv::Mat> split_mats;

    std::vector<float> dataVec(batch * height * width * depth);
    //std::cout <<batch<<"batch"<<conf.inputShape[0]<<std::endl;
    float *dataPtr = dataVec.data();

    for (const auto &image : images)
    {

        assert(image.rows == height);
        assert(image.cols == width);
        assert(image.type() == CV_8UC3 || image.type() == CV_32FC3);
        //image.convertTo(tmp, CV_32FC3);
        //image = tmp-cv::Scalar(meanb,meang,meanr)

        std::vector<cv::Mat> channels;
        split(image, channels);

        cv::Mat imageBlue(height, width, CV_32FC1, dataPtr);
        dataPtr += height * width;
        cv::Mat imageGreen(height, width, CV_32FC1, dataPtr);
        dataPtr += height * width;
        cv::Mat imageRed(height, width, CV_32FC1, dataPtr);
        dataPtr += height * width;

        channels.at(0).convertTo(imageBlue, CV_32FC1, xfactorb, -meanb * xfactorb);
        channels.at(1).convertTo(imageGreen, CV_32FC1, xfactorg, -meang * xfactorg);
        channels.at(2).convertTo(imageRed, CV_32FC1, xfactorr, -meanr * xfactorr);
    }
    return dataVec;
}

std::vector<cv::Mat> getinputimg(int num)
{
    std::vector<std::string> paths;
    for (int i = 0; i < num; ++i)
    { 
        paths.push_back("../ut/4354.jpg");
    }
    std::vector<cv::Mat> inputs;
    for (int i = 0; i < num; ++i)
    {
        auto img = cv::imread(paths[i]);
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(IMG_W, IMG_H), 0, 0);
        inputs.push_back(resized.clone());
    }
    return inputs;
}

static int init_num = 0;
void *mythread(void *threadid)
{
    int tid = *((int *)threadid);
    struct timeval start1;
    struct timeval end1;
    unsigned long timer;
    std::string modelfile = "../models/dcl_yt3.trt";
    std::cout << "modelfile: " << modelfile.size() << std::endl;
    auto hand = VehicleFgvcInstance(modelfile,
            tid % 4, small_batchsize, big_batchsize);
    init_num++;
    //read image
    while (init_num < NUM_THREADS)
    {
        usleep(1000);
    }
    auto inputs = getinputimg(big_batchsize);
    // ClassifyVehicleFgvc(hand, inputs); // Call CPU Version
    gettimeofday(&start1, NULL);
    VehicleFgvcResult RE;
    if (GPU_INPUT)
    {
        std::cout<<"main.cpp GPU_INPUT 1"<<std::endl;
        std::vector<float> input_src = _preProcess(inputs);
        std::cout<<"main.cpp GPU_INPUT 2 d="<<input_src.data()[8]<<";"<<std::endl;
        float *pGpu;
        void* deviceMem;
        (cudaMalloc(&deviceMem, input_src.size() * sizeof(float)));
        std::cout<<"main.cpp GPU_INPUT 3 size="<<input_src.size() * sizeof(float)<<";"<<std::endl;
        pGpu = (float*)deviceMem;
        cudaMemcpy(pGpu, input_src.data(),
                   input_src.size() * sizeof(float), cudaMemcpyHostToDevice);
        std::cout<<"main.cpp GPU_INPUT 4 copy_size="<<input_src.size() * sizeof(float)<<";"<<std::endl;
        // call DCL interface
        RE = ClassifyVehicleFgvc_GPU(hand, pGpu, small_batchsize); //获得检测结果
        std::cout << " gpu result is OK! v0.0.1" << std::endl;
        /*for (int i = 0; i < RE.CarNum; ++i)
        {
            std::cout << RE.headProb[i] << " " << std::endl;
        }*/
        cudaFree(pGpu);
    }
    std::cout<<"main.cpp: step 6"<<std::endl;
    gettimeofday(&end1, NULL);
    timer = 1000000 * (end1.tv_sec - start1.tv_sec) + end1.tv_usec - start1.tv_usec;
    if(GPU_DETECT_INPUT)
    {
        // Call other DCL interface
        std::cout << "GPU_DETECT_INPUT: " << std::endl;
        std::vector<cv::Mat> gpu_img ;
        std::vector<unsigned char*> cudaSrc;
        std::vector<int> srcWidth;
        std::vector<int> srcHeight;
//        std::vector<float > mean = {0,2,1};
//        std::vector<float > std = {1,2,3};
//        mean = {0.485, 0.485, 0.485};
//        std = {0.225, 0.225, 0.225};
        /*std::vector<ITS_Vehicle_Result_Detect> cpuDetect(inputs.size());
        for (int t = 0; t < inputs.size(); ++ t)
        {
            cv::Mat m = cv::Mat::zeros(549, 549, CV_8UC3);
            //cv::Rect rect(0, 0, 224, 224);  
            cv::Rect rect(0, 0, IMG_W, IMG_H);  
            //cv::Rect rect2(224, 224, 224, 224);  
            cv::Rect rect2(IMG_W, IMG_H, IMG_W, IMG_H);  
            inputs[t].copyTo(m(rect));
            inputs[t].copyTo(m(rect2));
            int imgSize = m.step[0]*m.rows;
			auto & img = m;
            assert(imgSize = 3*img.cols*img.rows);
            void*pgpu;
            cudaMalloc((void**)&pgpu, imgSize);
            cudaSrc.push_back((unsigned char *)pgpu);
            cudaMemcpy(pgpu,img.data,imgSize,cudaMemcpyHostToDevice);
            srcWidth.push_back(img.cols);
            srcHeight.push_back(img.rows);
            initDet(cpuDetect[t],5);
        }
        std::vector<Car_HEAD_TAIL_Result> all_results = 
        Detect_CarHeadAndTail_FromDetectGPU(hand,
                                    cudaSrc,
                                    srcWidth, srcHeight,
                                    cpuDetect);
        for (int u = 0; u < all_results.size(); ++u)
        {
            auto &RE = all_results[u];
            for (int i = 0; i < RE.CarNum; ++i)
            {
                std::cout <<u<<" " <<RE.headProb[i] << " " << std::endl;
            }
            std::cout <<std::endl;
        }*/
        std::cout<<"Call other DCL interface is OK"<<std::endl;
        

    }
    ReleaseVehicleFgvcInstance(hand);
}

int main()
{
    clock_t start, end;
    pthread_t threads[NUM_THREADS];
    int indexes[NUM_THREADS]; // 用数组来保存i的值
    for (int i = 0; i < NUM_THREADS; i++)
    {
        indexes[i] = i; //先保存i的值
        // 传入的时候必须强制转换为void* 类型，即无类型指针
        int rc = pthread_create(&threads[i], NULL, mythread, (void *)&(indexes[i]));
        if (rc)
        {
            std::cout << "Error:无法创建线程," << rc << std::endl;
            exit(-1);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
        //cout << "right:创建线程," << i << endl;
    }
    printf("end\n");
    //ITS_VehicleRecRelease(pInstance0);
    return 0;
}
