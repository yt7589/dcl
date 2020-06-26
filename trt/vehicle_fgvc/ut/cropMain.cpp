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
#include <codecvt>
#define NUM_THREADS 1

const int DEPTH = 3;
const int IMG_W = 224;
const int IMG_H = 224;
int big_batchsize = 8;
int small_batchsize = 8;
const bool GPU_INPUT = true;
const bool GPU_DETECT_INPUT = true;
// 图片预处理相关，定义见config.py第17行
const float MEAN_R = 0.485 * 255;
const float MEAN_G = 0.456 * 255;
const float MEAN_B = 0.406 * 255;
const float XFACTOR_R = 1 / (0.229 * 255);
const float XFACTOR_G = 1 / (0.224 * 255);
const float XFACTOR_B = 1 / (0.225 * 255);

int FBLOCK_MAX_BYTES = 1024;
char *szBuf;
void Split(const std::string& src, const std::string& separator, std::vector<std::string>& dest);
vector<vector<string>> GetTestDsSamples();

void  initDet(ITS_Vehicle_Result_Detect &det,const int detNum){
    det.CarNum = detNum;
    for(int i=0; i< detNum; ++i) {
        det.iLeft[i] =100;
        det.iTop[i] = 100;
        det.iRight[i] = 800; // 224
        det.iBottom[i] = 800; //224
    }
}

/**
 * 
 */
std::vector<cv::Mat> GetInputImg(vector<vector<string>> samples, int num)
{
    cv::Mat img;
    size_t imgSize;
    std::vector<cv::Mat> inputs;
    for (int t = 0; t < num; ++ t)
    {
        img = cv::imread(samples[t][0]);
        std::cout<<"img: "<<samples[t][0]<<"; classId: "<<samples[t][1]<<"; !!!!"<<std::endl;
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(IMG_W, IMG_H), 0, 0);
        inputs.push_back(resized.clone());
    }
    return inputs;
}

std::vector<float> PreProcess(const std::vector<cv::Mat> &images)
{
    auto batch = images.size(); //_config.inputShape[0];
    //assert(batch <= conf.inputShape[0]);
    auto depth = DEPTH;
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
        channels.at(0).convertTo(imageBlue, CV_32FC1, XFACTOR_B, -MEAN_B * XFACTOR_B);
        channels.at(1).convertTo(imageGreen, CV_32FC1, XFACTOR_G, -MEAN_G * XFACTOR_G);
        channels.at(2).convertTo(imageRed, CV_32FC1, XFACTOR_R, -MEAN_R * XFACTOR_R);
    }
    return dataVec;
}



static int init_num = 0;
void *mythread(void *threadid)
{
    int tid = *((int *)threadid);
    struct timeval start1;
    struct timeval end1;
    unsigned long timer;
    std::string modelfile = "/media/zjkj/35196947-b671-441e-9631-6245942d671b/"
                            "yantao/fgvc/dcl/trt/vehicle_fgvc/models/dcl_v009.trt";
    auto hand = VehicleFgvcInstance(modelfile,
            tid % 4, small_batchsize, big_batchsize);
    // 获取测试数据集上样本
    vector<vector<string>> samples = GetTestDsSamples();

    // Call other DCL interface
    std::cout << "GPU_DETECT_INPUT: " << std::endl;
    int batchSize = 8;

    auto inputs = GetInputImage(samples, batchsize);
    std::vector<float> input_src = PreProcess(inputs);
    float *pGpu;
    void* deviceMem;
    cudaMalloc(&deviceMem, input_src.size() * sizeof(float));
    pGpu = (float*)deviceMem;
    cudaMemcpy(pGpu, input_src.data(),
                   input_src.size() * sizeof(float), cudaMemcpyHostToDevice);


    std::vector<cv::Mat> gpu_img ;
    std::vector<unsigned char*> cudaSrc;
    std::vector<int> srcWidth;
    std::vector<int> srcHeight;
    std::vector<ITS_Vehicle_Result_Detect> cpuDetect(inputs.size());
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



    clock_t classify_start, classify_end;
    classify_start = clock();
    auto all_results = ClassifyVehicleFgvcFromDetectGPU(hand,
                                cudaSrc,srcWidth, srcHeight,
                                cpuDetect);
    classify_end = clock();
    std::cout<<"程序运行时间："<<((double)(classify_end - classify_start))/CLOCKS_PER_SEC*1000.0<<"毫秒;"<<std::endl;
    for (int u = 0; u < all_results.size(); ++u)
    {
        auto &RE = all_results[u];
        std::cout<<RE.tempResult[0].tempVehicleType<<"; classId="<<RE.tempResult[0].iVehicleSubModel<<std::endl;
        /*for (int i = 0; i < RE.iNum; ++i)
        {

            std::cout << " batch: " << u
                        << " car: "<< i
                        <<" conf :" <<RE.tempResult[i].fConfdence
                        <<" clsID :" <<RE.tempResult[i].iVehicleSubModel
                        << " " << std::endl;
        }
        std::cout <<std::endl;*/
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

vector<vector<string>> GetTestDsSamples()
{
    szBuf = (char*)malloc(FBLOCK_MAX_BYTES * sizeof(char) + 1);
    std::string strFileUTF8 = "/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/datasets/CUB_200_2011/anno/test_ds_v4.txt";
    std::string strFileName = strFileUTF8;
    memset(szBuf, 0, sizeof(char) * FBLOCK_MAX_BYTES);
    std::string strMessage;
    FILE * fp = NULL;
    fp = fopen(strFileName.c_str(), "rb");
    if (fp != NULL)
    {
        // fseek(fp, sizeof(char) * 3, 0);
        while(fread(szBuf, sizeof(char), FBLOCK_MAX_BYTES, fp) > 0)
        {
            szBuf[FBLOCK_MAX_BYTES] = '\0';
            strMessage += szBuf;
            memset(szBuf, 0, sizeof(char) * FBLOCK_MAX_BYTES);
        }
    }
    fclose(fp);
    vector<string> lines;
    vector<string> sample;
    vector<vector<string>> samples;
    Split(strMessage, "\n", lines);
    vector<string>::iterator iter;
    string line;
    for (iter=lines.begin(); iter!=lines.end(); iter++)
    {
        line = *iter;
        if (line.length() > 10) {
            Split(line, "*", sample);
            samples.push_back(sample);
        }
    }
    return samples;
}

void Split(const std::string& src, const std::string& separator, std::vector<std::string>& dest) //字符串分割到数组
{
 
        //参数1：要分割的字符串；参数2：作为分隔符的字符；参数3：存放分割后的字符串的vector向量
 
	string str = src;
	string substring;
	string::size_type start = 0, index;
	dest.clear();
	index = str.find_first_of(separator,start);
	do
	{
		if (index != string::npos)
		{    
			substring = str.substr(start,index-start );
			dest.push_back(substring);
			start =index+separator.size();
			index = str.find(separator,start);
			if (start == string::npos) break;
		}
	}while(index != string::npos);
 
	//the last part
	substring = str.substr(start);
	dest.push_back(substring);
}
