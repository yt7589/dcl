#include "vehicle_fgvc.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "api_global.hpp"
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
void *VehicleFgvcInstance(string modelpath,
                             int cardnum, int max_batch_size,
                             int max_big_pic,
                             string businessType) //端口初始化
{
    //assert(max_batch_size == 32);
    std::cout<<"VehicleFgvcInstance 1"<<std::endl;
    std::lock_guard<std::mutex> lock(my_lock);
    std::cout<<"VehicleFgvcInstance 2"<<std::endl;
    auto eng =  (PredictorAPI*)(Reflector::Instance().
            CreateObject(businessType));//new VTNPredictorAPI();
    std::cout<<"VehicleFgvcInstance 3"<<std::endl;
    InputConfig iconfig;
    iconfig.devices.push_back(cardnum);
    std::cout<<"VehicleFgvcInstance 4"<<std::endl;
    if (modelpath.find("engine") != modelpath.npos)
    {
        std::cout<<"VehicleFgvcInstance 4.1 model type is engine"<<std::endl;
        iconfig.modelType = "engine";
    }
    else if (modelpath.find("onnx") != modelpath.npos)
    {
        std::cout<<"VehicleFgvcInstance 4.2 model type is onnx"<<std::endl;
        iconfig.modelType = "onnx";
    }
    else
    {
        std::cout<<"VehicleFgvcInstance 4.3 model type set to default engine"<<std::endl;
        iconfig.modelType = "engine";
    }
    std::cout<<"VehicleFgvcInstance 5"<<std::endl;

    iconfig.maxBatchSize = max_batch_size;

    if (eng->init(std::vector<std::string> {modelpath},
                  iconfig))
    {
        std::cout<<"VehicleFgvcInstance 6"<<std::endl;

        G_SOURCE[(void *)eng] = eng; //todo CHECK EXIST
        GInfo tmp;
        tmp.cardnum = cardnum;
        tmp.max_batch_size = max_batch_size;
        std::cout<<"VehicleFgvcInstance 7"<<std::endl;
        tmp.tempCudaDet = initTempCudaDet(cardnum, 8); // std::vector<uchar *> cudaSrc // oriBatchSize =cudaSrc.size()
        tmp.cudaCropImages = initCropAndResizeImages(cardnum, 8, MAX_CAR_NUM, IMG_W, IMG_H);
        std::cout<<"VehicleFgvcInstance 8"<<std::endl;
        //max_big_pic+1  zuihou yige yongyu qianxiang shuru zhongzhuan
        G_GInfo[(void *)eng] = tmp;
        std::cout<<"VehicleFgvcInstance 9"<<std::endl;
        return (void *)eng;
    }
    else
    {
        std::cout<<"VehicleFgvcInstance 100"<<std::endl;
        delete eng;
    }
    return nullptr;
}

//====================================  =============================================
//Car_HEAD_TAIL_Result Detect_CarHeadAndTail(void *iInstanceId, std::vector<cv::Mat> imgs) //获得检测结果
VehicleFgvcResult ClassifyVehicleFgvc(void *iInstanceId, std::vector<cv::Mat> imgs)
{
    VehicleFgvcResult result;
    auto it = G_SOURCE.find(iInstanceId);
    if (it != G_SOURCE.end())
    {
        std::vector<std::vector<float>> out_results;
        it->second->forward(imgs, out_results);
        //result.CarNum = out_results.size();
        result.vid = out_results.size();
        std::cout<<"picture_num="<<out_results.size()<<std::endl;
        for (int i = 0; i < out_results.size(); ++i)
        {
            //result.headProb[i] = (out_results[i][1]);
            for (int j=0; j<out_results[i].size(); j++)
            {
                std::cout<<out_results[i][j]<<", ";
            }
            std::cout<<std::endl;
        }
    }
    else
    {
        assert(false); //TODO
    }
    return result;
}

//Car_HEAD_TAIL_Result Detect_CarHeadAndTail_GPU(void *iInstanceId, float *pGpuData, int num) //获得检测结果
VehicleFgvcResult ClassifyVehicleFgvc_GPU(void *iInstanceId, float *pGpuData, int num)
{
    assert(num > 0);
    std::cout<<"ClassifyVehicleFgvc_GPU 1:num="<<num<<";"<<std::endl;
    VehicleFgvcResult result;
    auto it = G_SOURCE.find(iInstanceId);
    std::cout<<"ClassifyVehicleFgvc_GPU 2"<<std::endl;
    int max_idx = -1;
    double max_val = -0.1;
    if (it != G_SOURCE.end())
    {
        std::cout<<"ClassifyVehicleFgvc_GPU 3"<<std::endl;
        std::vector<std::vector<float>> out_results;
        it->second->forward(pGpuData, num, out_results);
        std::cout<<"ClassifyVehicleFgvc_GPU 4"<<std::endl;
        //result.CarNum = out_results.size();
        int picture_num = out_results.size();
        int class_num = 0;
        std::cout<<"picture_num="<<picture_num<<std::endl;
        for (int i = 0; i < picture_num; ++i)
        {
            max_idx = -1;
            max_val = -1.0;
            //result.headProb[i] = (out_results[i][1]);
            class_num = out_results[i].size();
            for (int j=0; j<class_num; j++)
            {
                if (out_results[i][j] > max_val) {
                    max_idx = j;
                    max_val = out_results[i][j];
                }
            }
            std::cout<<"###"<<i+1<<":"<<max_idx<<"("<<max_val<<");"<<std::endl;
        }
    }
    else
    {
        assert(false); //TODO
    }
    return result;
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

/*std::vector<Car_HEAD_TAIL_Result>
Detect_CarHeadAndTail_FromDetectGPU(void *iInstanceId,
                                    std::vector<unsigned char *> &cudaSrc,
                                    std::vector<int> &srcWidth, std::vector<int> &srcHeight,
                                    std::vector<ITS_Vehicle_Result_Detect> &cpuDetect) //获得检测结 */
std::vector<VehicleFgvcResult> ClassifyVehicleFgvcFromDetectGPU(void *iInstanceId,
                                    std::vector<unsigned char *> &cudaSrc,
                                    std::vector<int> &srcWidth, std::vector<int> &srcHeight,
                                    std::vector<ITS_Vehicle_Result_Detect> &cpuDetect)
{
    return std::vector<VehicleFgvcResult>();
}

//int ReleaseSDK_CarHeadAndTail(void *iInstanceId) //释放接口
int ReleaseVehicleFgvcInstance(void *iInstanceId)
{

    auto it_info = G_GInfo.find(iInstanceId);

    if (it_info != G_GInfo.end())
    {

        cudaFree(it_info->second.tempCudaDet);

        for (auto &i : it_info->second.cudaCropImage)
        {
            cudaFree(i);
        }
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
