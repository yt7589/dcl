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
    std::vector<float *> cudaCropImage;
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
        tmp.tempCudaDet = initTempCudaDet(cardnum, MAX_CAR_NUM);
        //tmp.cudaCropImage = initCropAndResizeImages(cardnum, max_big_pic+1, MAX_CAR_NUM, 224, 224);
        tmp.cudaCropImage = initCropAndResizeImages(cardnum, max_big_pic+1, MAX_CAR_NUM, 448, 448);
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
            std::cout<<"###"<<i+1<<":";
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
    VehicleFgvcResult result;
    auto it = G_SOURCE.find(iInstanceId);
    if (it != G_SOURCE.end())
    {
        std::vector<std::vector<float>> out_results;
        it->second->forward(pGpuData, num, out_results);
        //result.CarNum = out_results.size();
        result.vid = out_results.size();
        std::cout<<"picture_num="<<out_results.size()<<std::endl;
        for (int i = 0; i < out_results.size(); ++i)
        {
            //result.headProb[i] = (out_results[i][1]);
            std::cout<<"###"<<i+1<<":";
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
//    for (auto iter = cpuDetect.begin(); iter != cpuDetect.end(); ++iter)
//    {
//        int num = iter->CarNum;
//        std::cout << " " << num << " "<<std::endl;
//    }

    auto it = G_GInfo.find(iInstanceId);
    int max_batch_size;
    int cardnum;
    ITS_Vehicle_Result_Detect *tempCudaDet;
    std::vector<float *> cudaCropImage;

    if (it != G_GInfo.end())
    {
        cardnum = it->second.cardnum;
        max_batch_size = it->second.max_batch_size;
        tempCudaDet = it->second.tempCudaDet;
        cudaCropImage = it->second.cudaCropImage;
    }
    else
    {
        assert(false);
        return std::vector<VehicleFgvcResult>();
    }

    //cv::Mat img = cv::imread("/home/ubuntu/ds/lzy/tiny_tensorrt_vtn/test_pic/int8.jpg");
    //size_t imgSize = img.step[0]*img.rows;
    int batchsize = cudaSrc.size(),
        maxOutWidth = 224, maxOutHeight = 224;
    assert(batchsize <= 8);
    if (batchsize > 8)
    {
        return std::vector<VehicleFgvcResult>();
    }
    //std::vector<ITS_Vehicle_Result_Detect> cpuDetect(batchsize);
    //initDet(cpuDetect[0],50);
    assert(batchsize == cpuDetect.size());
    assert(batchsize == srcWidth.size());
    assert(batchsize == srcHeight.size());
    std::vector<float> mean = {0, 0, 0};
    std::vector<float> std = {1, 1, 1};
    bool test_write = false;
    if (!test_write)
    {
        mean = {0.485, 0.485, 0.485};
        std = {0.225, 0.225, 0.225};
    }
    int dstW = 224, dstH = dstW;
    nvHTCropAndReizeLaunch(cudaCropImage, cudaSrc, cpuDetect, tempCudaDet, srcWidth, srcHeight, mean, std, batchsize, dstW, dstH);
    std::vector<cv::Mat> out_img;
    std::vector<float > dst(dstW*dstH*3);
    for (int b = 0; b < batchsize && test_write ;++b){
        int detnum = cpuDetect[b].CarNum;
        for(int j=0;j<1;++j){
            std::vector<cv::Mat> dst_channels;
            for (int i = 0; i < 3; ++i) {
                cv::Mat dst_channel(dstW,dstH,CV_32FC1);
                cudaMemcpy(dst_channel.data,cudaCropImage[b] + j*dstW*dstH*3 + i*dstW*dstH  ,sizeof(float)*dstW*dstH*1,cudaMemcpyDeviceToHost);
                dst_channels.push_back(dst_channel);
        }
        cv::Mat target(dstH,dstW,CV_32FC3);
        cv::merge(dst_channels,target);
        cv::Mat out;
        target.convertTo(out,CV_8UC3,255);
        cv::imwrite("ab_test.jpg",out);
        }
    }
    std::vector<OneBatch> totaldata(1);
    for (int b = 0; b < batchsize; ++b)
    {
        assert(totaldata.back().pic_num <= max_batch_size);
        if (totaldata.back().pic_num == max_batch_size)
        {
            totaldata.emplace_back(OneBatch());
        }

        int detnum = cpuDetect[b].CarNum;
        int left = detnum;
        int start = 0;
        while (left > 0)
        {
            int needsize = max_batch_size - totaldata.back().pic_num;
            if (left <= needsize)
            {
                OnePic onepic;
                onepic.batch_index = b;
                onepic.start = start;
                onepic.end = start + left;
                totaldata.back().one_batch.push_back(onepic);
                totaldata.back().pic_num += left;
                //left = 0;
                break;
                ///进入下一张大图片
            }
            else
            {
                OnePic onepic;
                onepic.batch_index = b;
                onepic.start = start;
                onepic.end = start + needsize;
                start += needsize;
                left -= needsize;
                totaldata.back().one_batch.push_back(onepic);
                totaldata.back().pic_num += needsize;
                totaldata.emplace_back(OneBatch());
            }
        }
    }
    std::vector<float> tmpresult;
    /*
    class OnePic
    {
        public:
        OnePic():start(0),end(0){}
        int batch_index;  //0-2:  0,1
        int start;
        int end;
    };
    class OneBatch
    {
        public:
        OneBatch():pic_num(0){}
        std::vector<tm> one_batch;
        int pic_num;
    }*/
    if (totaldata.back().pic_num == 0)
    {
        totaldata.pop_back();
    }
    for (int t = 0; t < totaldata.size(); t++)
    {
        auto &onebatch = totaldata[t];
        float *batchdata = cudaCropImage.back();
        int copyednum = 0;
        for (int picindex = 0; picindex < onebatch.one_batch.size(); ++picindex)
        {
            auto &onepic = onebatch.one_batch[picindex];
            cudaMemcpy(batchdata + copyednum * 224 * 224 * 3, cudaCropImage[onepic.batch_index] + onepic.start * 224 * 224 * 3,
                       sizeof(float) * 224 * 224 * 3 * (onepic.end - onepic.start), cudaMemcpyDeviceToDevice);
            copyednum += onepic.end - onepic.start;
        }
        //std::cout << t << " copyednum : " << copyednum << " " << std::endl;
        assert(copyednum == onebatch.pic_num);
        VehicleFgvcResult re = ClassifyVehicleFgvc_GPU(iInstanceId, batchdata, copyednum); //获得检测结果
        /*assert(re.CarNum == copyednum);
        for (int m = 0; m < re.CarNum; ++m)
        {
            tmpresult.push_back(re.headProb[m]);
        }*/
    }
    std::vector<VehicleFgvcResult> final_result(batchsize);
    float *curptr = tmpresult.data();
    for (int b = 0; b < batchsize; ++b)
    {
        /*
        int detnum = cpuDetect[b].CarNum;
        final_result[b].CarNum = cpuDetect[b].CarNum;
        memcpy(final_result[b].headProb, curptr, sizeof(float) * final_result[b].CarNum);
        curptr += final_result[b].CarNum;
        */
       std::cout<<"get and parse the result ?????"<<std::endl;
    }
    assert(curptr == tmpresult.data() + tmpresult.size());
    return final_result;
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
