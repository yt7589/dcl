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
    //assert(max_batch_size == 32);
    std::lock_guard<std::mutex> lock(my_lock);
    auto eng =  (PredictorAPI*)(Reflector::Instance().
            CreateObject(businessType));//new VTNPredictorAPI();
    InputConfig iconfig;
    iconfig.devices.push_back(cardnum);
    if (modelpath.find("engine") != modelpath.npos)
        iconfig.modelType = "engine";
    else if (modelpath.find("onnx") != modelpath.npos)
        iconfig.modelType = "onnx";
    else
    {
        iconfig.modelType = "engine";
    }

    iconfig.maxBatchSize = max_batch_size;

    if (eng->init(std::vector<std::string> {modelpath},
                  iconfig))
    {

        G_SOURCE[(void *)eng] = eng; //todo CHECK EXIST
        GInfo tmp;
        tmp.cardnum = cardnum;
        tmp.max_batch_size = max_batch_size;
        tmp.tempCudaDet = initTempCudaDet(cardnum, 8);
        tmp.cudaCropImages = initCropAndResizeImages(cardnum,
                max_big_pic+1, 8, 224, 224);
        //max_big_pic+1  zuihou yige yongyu qianxiang shuru zhongzhuan
        G_GInfo[(void *)eng] = tmp;
        return (void *)eng;
    }
    else
        delete eng;
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
//VehicleFgvcResult ClassifyVehicleFgvc_GPU(void* iInstanceId, float* pGpuData, int num)
//{
//    assert(num > 0);
//    Car_HEAD_TAIL_Result result;
//    auto it = G_SOURCE.find(iInstanceId);
//    if (it != G_SOURCE.end())
//    {
//        std::vector<std::vector<float>> out_results;
//
//        it->second->forward(pGpuData, num, out_results);
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
//    return result;
//}

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

/*std::vector<Car_HEAD_TAIL_Result>
Detect_CarHeadAndTail_FromDetectGPU(void *iInstanceId,
                                    std::vector<unsigned char *> &cudaSrc,
                                    std::vector<int> &srcWidth, std::vector<int> &srcHeight,
                                    std::vector<ITS_Vehicle_Result_Detect> &cpuDetect) //获得检测结果 */
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

    int batchsize = cudaSrc.size(),
        maxOutWidth = 224, maxOutHeight = 224;

    assert(batchsize == cpuDet.size());
    assert(batchsize == srcWidth.size());
    assert(batchsize == srcHeight.size());

    std::vector<float> mean = {0.485, 0.485, 0.485};
    std::vector<float> std = {0.225, 0.225, 0.225};

    nvHTCropAndReizeLaunch(cudaCropImages, cudaSrc, cpuDet,
            tempCudaDet, srcWidth, srcHeight,
            mean, std, batchsize, maxOutWidth, maxOutHeight);



    std::vector<Type_Vehicle_Result> final_result(batchsize);

    return std::move(final_result);
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
