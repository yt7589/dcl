#ifndef __VEHICLE_FGVC_H__
#define __VEHICLE_FGVC_H__
#include <string>
#include <vector>
using namespace std;
 #include "opencv2/core/core.hpp"

/*
typedef struct
{
 bool CarNum;                //0  HEAD   1  TAIL

}Car_HEAD_TAIL_Result;
*/

#include "AllModuleInclude.h"

#ifndef HEAD_TAIL_NUM
#define HEAD_TAIL_NUM 100
typedef struct
{
    int CarNum;
    float headProb[HEAD_TAIL_NUM];//bigger that 
} Car_HEAD_TAIL_Result;
#endif

typedef struct 
{
    int vid;
    char msg[100];
} VehicleFgvcResult;


#define JNADLL extern "C" __attribute__((visibility("default")))

//JNADLL void *CarHeadAndTailInstance(string modelpath, int GPU_ID, int max_batch_size, int max_big_pic = 8, std::string businessType = "task_head_tail"); //端口初始化
JNADLL void *VehicleFgvcInstance(string modelpath, int GPU_ID, 
            int max_batch_size, int max_big_pic = 8, 
            std::string businessType = "task_vehicle_fgvc"); //端口初始化

//====================================  =============================================
//JNADLL Car_HEAD_TAIL_Result Detect_CarHeadAndTail(void *iInstanceId, std::vector<cv::Mat> imgs); //获得检测结果
// get the result of vechicle FGVC
JNADLL VehicleFgvcResult ClassifyVehicleFgvc(void* iInstanceId, std::vector<cv::Mat> imgs);


//num: tupiao shuliang (3*224*224)
//pGpuData: bchw float   buffsize = num *c*h*w*sizeof(float)
//JNADLL Car_HEAD_TAIL_Result Detect_CarHeadAndTail_GPU(void *iInstanceId, float *pGpuData, int num); //获得检测结果
// 
JNADLL VehicleFgvcResult ClassifyVehicleFgvc_GPU(void* iInstanceId, float* pGpuData, int num);

/*JNADLL std::vector<Car_HEAD_TAIL_Result>
Detect_CarHeadAndTail_FromDetectGPU(void *iInstanceId,
                                    std::vector<unsigned char *> &cudaSrc,
                                    std::vector<int> &srcWidth, std::vector<int> &srcHeight,
                                    std::vector<ITS_Vehicle_Result_Detect> &cpuDet);*/
JNADLL std::vector<VehicleFgvcResult> ClassifyVehicleFgvcFromDetectGPU(void *iInstanceId,
                                    std::vector<unsigned char *> &cudaSrc,
                                    std::vector<int> &srcWidth, std::vector<int> &srcHeight,
                                    std::vector<ITS_Vehicle_Result_Detect> &cpuDet);

//JNADLL int ReleaseSDK_CarHeadAndTail(void *iInstanceId); //释放接口
JNADLL int ReleaseVehicleFgvcInstance(void* iInstanceId);

#endif
