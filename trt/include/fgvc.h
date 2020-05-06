#ifndef __FGVC_H__
#define __FGVC_H__

#include <string>

using namespace std;
//      "/home/zjkj/opencv347/opencv/opencv-3.4.7/modules/core/include/opencv2/core.hpp"
//#include <opencv2/core.hpp>
#include "/home/zjkj/opencv347/opencv/opencv-3.4.7/modules/core/include/opencv2/core.hpp"
#include <vector>
#include "AllModuleInclude.h"

typedef struct {
    std::vector<int> topKPINPAI{}; // top K的预测结果
    std::vector<float> topKConfidence{}; // top K对应的置信度
    int PINPAI;                //品牌对应的ID
    float confidence;     // 置信度
    char features[12000]; // 特征值
    int featureNums; // 特征值的个数

} CAR_FEATURE_RESULT;

#define JNADLL extern "C" __attribute__((visibility("default")))
//初始化的时候送进去模型的路径，内部在该目录下读取模型，max_batch_size表示一次批处理最大的小图数量
JNADLL void *
VehicleFeatureInstance(const string &modelPath, int GPU_ID, int max_batch_size);//端口初始化,参数1为模型路径，2为gpu卡号，3为最大batchsize


/*需在内部实现resize，和crop，送进去原图，和检测框的位置，输出结果，
const std::vector<unsigned char *> &   原图大图的闪存地址向量集
const std::vector<int> &,  原图大图的宽度
const std::vector<int> &,  原图大图的高度
std::vector<ITS_Vehicle_Result_Detect> &cpuDet); 原图中每个检测目标车辆的位置，最大为20个车辆，如果当前送入的车辆图片数量大于最大设置的批处理图片数量，
内部需根据实际图片数量多次调用批处理函数，并将结果送出
*/
JNADLL std::vector<std::vector<CAR_FEATURE_RESULT>> GetCarFeatureGPU(void *iInstanceId,
                                                     const std::vector<unsigned char *> &,
                                                     const std::vector<int> &,
                                                     const std::vector<int> &,
                                                     std::vector<ITS_Vehicle_Result_Detect> &cpuDet);

JNADLL int ReleaseSDKFeature(void *iInstanceId);//释放接口
JNADLL std::vector<float> parseFEATUREVector(const CAR_FEATURE_RESULT &toParse);
#endif