//#include "vehicle.h"
#include <pthread.h>
//#include <sys/time.h>
#include <time.h>
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
        det.iLeft[i] =100;
        det.iTop[i] = 100;
        det.iRight[i] = 800; // 224
        det.iBottom[i] = 800; //224
    }
}
static int init_num = 0;
void *mythread(void *threadid)
{
    int tid = *((int *)threadid);
    struct timeval start1;
    struct timeval end1;
    unsigned long timer;
    std::string modelfile = "/media/zjkj/35196947-b671-441e-9631-6245942d671b/"
                            "yantao/fgvc/dcl/trt/vehicle_fgvc/models/dcl_v002_cao.trt";
    std::cout << "modelfile: " << modelfile.size() << std::endl;
    auto hand = VehicleFgvcInstance(modelfile,
            tid % 4, small_batchsize, big_batchsize);

    // Call other DCL interface
    std::cout << "GPU_DETECT_INPUT: " << std::endl;
    int batchSize = 8;
    std::vector<unsigned char*> cudaSrc(batchSize);
    std::vector<int> srcWidth(batchSize);
    std::vector<int> srcHeight(batchSize);
    std::vector<ITS_Vehicle_Result_Detect> cpuDetect(cudaSrc.size());
    cv::Mat img  = cv::imread("../ut/1.jpg");
    size_t imgSize = img.step[0]*img.rows;
    for (int t = 0; t < cudaSrc.size(); ++ t)
    {
        cudaMalloc((void**)&(cudaSrc[t]), imgSize);
        cudaMemcpy(cudaSrc[t],img.data,imgSize,cudaMemcpyHostToDevice);
        srcWidth[t] = (img.cols);
        srcHeight[t] = (img.rows);
        initDet(cpuDetect[t],7);
    }
    clock_t classify_start, classify_end;
    classify_start = clock();
    auto all_results = ClassifyVehicleFgvcFromDetectGPU(hand,
                                cudaSrc,srcWidth, srcHeight,
                                cpuDetect);
    classify_end = clock()*1.0;
    std::cout<<"程序运行时间："<<(classify_end - classify_start)/CLOCKS_PER_SEC*1000<<"毫秒"<<std::endl;
    for (int u = 0; u < all_results.size(); ++u)
    {
        auto &RE = all_results[u];
        for (int i = 0; i < RE.iNum; ++i)
        {

            std::cout << " batch: " << u
                        << " car: "<< i
                        <<" conf :" <<RE.tempResult[i].fConfdence
                        <<" clsID :" <<RE.tempResult[i].iVehicleSubModel
                        << " " << std::endl;
        }
        std::cout <<std::endl;
    }
    std::cout<<"Call other DCL interface is OK"<<std::endl;

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
