#include "vehicle_fgvc.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "predictor_api.hpp"
#include "reflect.hpp"
#include <mutex> /*std::mutex、 std::lock_guard*/
#include "cuda_runtime.h"
std::mutex my_lock;
#include "nvHTCropAndResize.h"
#include <map>
#include <time.h>

class GInfo
{
public:
    int cardnum;
    int max_batch_size;
    ITS_Vehicle_Result_Detect *tempCudaDet;
    float * cudaCropImages;
};
std::map<void *, PredictorAPI *> G_SOURCE;
std::map<void *, GInfo> G_GInfo;
// 颜色均值和方差，需要与config.py中的定义一致
std::vector<float> g_rgb_mean = {0.485, 0.456, 0.406};
std::vector<float> g_rgb_std = {0.229, 0.224, 0.225};
// 性能测试相关全局变量定义
double g_total_run_time = 0.0;
int g_total_operation = 0;

/*void *CarHeadAndTailInstance(string modelpath,
                             int cardnum, int max_batch_size,
                             int max_big_pic,
                             string businessType) //端口初始化
*/
void *VehicleFgvcInstance(string modelpath,
                             int cardnum, int max_batch_size,
                             int max_big_pic,
                             string businessType) //端口初始化
{
    std::cout<<"VehicleFgvcInstance::VehicleFgvcInstance 1"<<std::endl;
    //assert(max_batch_size == 32);
    std::lock_guard<std::mutex> lock(my_lock);
    auto eng =  (PredictorAPI*)(Reflector::Instance().
            CreateObject(businessType));//new VTNPredictorAPI();
    std::cout<<"VehicleFgvcInstance::VehicleFgvcInstance 2"<<std::endl;
    InputConfig iconfig;
    iconfig.devices.push_back(cardnum);
    std::cout<<"VehicleFgvcInstance::VehicleFgvcInstance 3"<<std::endl;
    if (modelpath.find("engine") != modelpath.npos)
        iconfig.modelType = "engine";
    else if (modelpath.find("onnx") != modelpath.npos)
    {
        std::cout<<"##### load from onnx file"<<std::endl;
        iconfig.modelType = "onnx";
    }
    else
    {
        iconfig.modelType = "engine";
    }
    std::cout<<"VehicleFgvcInstance::VehicleFgvcInstance 4"<<std::endl;

    iconfig.maxBatchSize = max_batch_size;

    if (eng->init(std::vector<std::string> {modelpath},
                  iconfig))
    {
        std::cout<<"VehicleFgvcInstance::VehicleFgvcInstance 5"<<std::endl;
        G_SOURCE[(void *)eng] = eng; //todo CHECK EXIST
        GInfo tmp;
        tmp.cardnum = cardnum;
        tmp.max_batch_size = max_batch_size;
        std::cout<<"VehicleFgvcInstance::VehicleFgvcInstance 6"<<std::endl;
        tmp.tempCudaDet = initTempCudaDet(cardnum, max_batch_size);
        std::cout<<"VehicleFgvcInstance::VehicleFgvcInstance 7"<<std::endl;
        tmp.cudaCropImages = initCropAndResizeImages(cardnum,
                max_batch_size, MAX_CAR_NUM, 224, 224);
                std::cout<<"VehicleFgvcInstance::VehicleFgvcInstance 9"<<std::endl;
        //max_big_pic+1  zuihou yige yongyu qianxiang shuru zhongzhuan
        G_GInfo[(void *)eng] = tmp;
        return (void *)eng;
    }
    else
        delete eng;
    std::cout<<"VehicleFgvcInstance::VehicleFgvcInstance 10"<<std::endl;
    return nullptr;
}

//====================================  �����=============================================
//Car_HEAD_TAIL_Result Detect_CarHeadAndTail(void *iInstanceId, std::vector<cv::Mat> imgs) //获得检测结果
//VehicleFgvcResult ClassifyVehicleFgvc(void* iInstanceId, std::vector<cv::Mat> imgs)
//{
//    Car_HEAD_TAIL_Result result;
//    auto it = G_SOURCE.find(iInstanceId);
//    if (it != G_SOURCE.end())
//    {
//        std::vector<std::vector<float>> out_results;
//
//        it->second->forward(imgs, out_results);
//
//
//        result.CarNum = out_results.size();
//        for (int i = 0; i < out_results.size(); ++i)
//        {
//            result.headProb[i] = (out_results[i][1]);
//        }
//    }
//    else
//    {
//        assert(false); //TODO
//    }
//    return ;
//}

//Car_HEAD_TAIL_Result Detect_CarHeadAndTail_GPU(void *iInstanceId, float *pGpuData, int num) //获得检测结果
VehicleFgvcResult ClassifyVehicleFgvc_GPU(void* iInstanceId, float* pGpuData, int num)
{
    assert(num > 0);
    Car_HEAD_TAIL_Result result;
    auto it = G_SOURCE.find(iInstanceId);
    if (it != G_SOURCE.end())
    {
        std::vector<std::vector<float>> out_results;

        it->second->forward(pGpuData, num, out_results);

        result.CarNum = out_results.size();
        for (int i = 0; i < out_results.size(); ++i)
        {
            result.headProb[i] = (out_results[i][1]);
        }
    }
    else
    {
        assert(false); //TODO
    }
    return VehicleFgvcResult{};
}

class OnePic
{
public:
    OnePic() : start(0), end(0), batch_index(0) {}
    int batch_index; //0-2:  0,1
    int start;
    int end;
};
class OneBatch
{
public:
    OneBatch() : pic_num(0) {}
    std::vector<OnePic> one_batch;
    int pic_num;
};

std::vector<Type_Vehicle_Result> ClassifyVehicleFgvcFromDetectGPU(void *iInstanceId,
                                    std::vector<unsigned char *> &cudaSrc,
                                    std::vector<int> &srcWidth, std::vector<int> &srcHeight,
                                    std::vector<ITS_Vehicle_Result_Detect> &cpuDet)
{
    auto it = G_GInfo.find(iInstanceId);
    int max_batch_size;
    int cardnum;
    ITS_Vehicle_Result_Detect *tempCudaDet;
    float * cudaCropImages;
    auto predictor = G_SOURCE.find(iInstanceId);
    if (it != G_GInfo.end())
    {
        cardnum = it->second.cardnum;
        max_batch_size = it->second.max_batch_size;
        tempCudaDet = it->second.tempCudaDet;
        cudaCropImages = it->second.cudaCropImages;
    }
    else
    {
        assert(false);
        return std::vector<Type_Vehicle_Result>();
    }
    int batchsize = cudaSrc.size(), maxOutWidth = 224, maxOutHeight = 224;

    assert(batchsize == cpuDet.size());
    assert(batchsize == srcWidth.size());
    assert(batchsize == srcHeight.size());

    int carNum = nvHTCropAndReizeLaunch(cudaCropImages, cudaSrc, cpuDet,
            tempCudaDet, srcWidth, srcHeight,
            g_rgb_mean, g_rgb_std, batchsize, maxOutWidth, maxOutHeight);

    int batchTimes = carNum / max_batch_size;
    int lastPic = carNum % max_batch_size;
    if(predictor == G_SOURCE.end()){
        assert(false);
        return std::vector<Type_Vehicle_Result>();
    }
    std::vector<Type_Vehicle_Result> result(batchsize);
    int batchId = 0,curCarNum = 0;
    int imgSize = maxOutHeight*maxOutHeight*3;

    clock_t batch_start, batch_end;
    batch_start = clock();
    for(int i = 0; i< batchTimes; ++i) {
        std::vector<std::vector<float>> out_results;
        clock_t item_start, item_end;
        item_start = clock();
        predictor->second->forward(cudaCropImages + i*max_batch_size*imgSize,
                max_batch_size, out_results);
        item_end = clock();
        //std::cout<<"yt处理单个样本时间："<<((double)(item_end - item_start))/CLOCKS_PER_SEC*1000.0<<"毫秒;"<<std::endl;
        g_total_run_time += ((double)(item_end - item_start))/CLOCKS_PER_SEC*1000.0;
        g_total_operation++;
        for(int n = 0; n < max_batch_size; ++n){
            if(curCarNum == cpuDet[batchId].CarNum) {
                batchId++;
                curCarNum=0;
            }
            float conf=out_results[0][n];
            int clsId= (reinterpret_cast<int*>(out_results[1].data()))[n];
            result[batchId].iNum=curCarNum+1;
            result[batchId].tempResult[curCarNum].fConfdence = conf;
            result[batchId].tempResult[curCarNum].iVehicleSubModel = clsId;
            curCarNum++;
        }
    }
    if (lastPic > 0) {
        std::vector<std::vector<float>> out_results;
        predictor->second->forward(
                static_cast<float*>(cudaCropImages) + batchTimes*max_batch_size*imgSize,
                lastPic, out_results);
        for(int n = 0; n < lastPic; ++n){
            if(curCarNum == cpuDet[batchId].CarNum) {
                batchId++;
                curCarNum=0;
            }
            float conf=out_results[0][n];
            int clsId= reinterpret_cast<int*>(out_results[1].data())[n];
            result[batchId].iNum= curCarNum+1;
            result[batchId].tempResult[curCarNum].fConfdence = conf;
            result[batchId].tempResult[curCarNum].iVehicleSubModel = clsId;
            curCarNum++;
        }
    }
    batch_end = clock();
    std::cout<<"处理整个批次时间："<<((double)(batch_end - batch_start))/CLOCKS_PER_SEC*1000.0<<"毫秒;"<<std::endl;


    return std::move(result);
}

//int ReleaseSDK_CarHeadAndTail(void *iInstanceId) //释放接口
int ReleaseVehicleFgvcInstance(void* iInstanceId)
{

    auto it_info = G_GInfo.find(iInstanceId);

    if (it_info != G_GInfo.end())
    {

        cudaFree(it_info->second.tempCudaDet);
        cudaFree(it_info->second.cudaCropImages);
        G_GInfo.erase(it_info);
    }

    auto it = G_SOURCE.find(iInstanceId);
    if (it != G_SOURCE.end())
    {
        delete it->second;
        G_SOURCE.erase(it);
        return 1;
    }
    else
        return 0;
}
