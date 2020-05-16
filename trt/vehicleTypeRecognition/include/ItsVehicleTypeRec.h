#ifndef _ITS_VEHICLE_TYPE_REC_H_
#define _ITS_VEHICLE_TYPE_REC_H_
#include "inc.h" 
#define VTR_API extern "C" __attribute__((visibility("default"))) 





typedef struct
{	
	char tempVehicleType[1024];    //车辆类型--总名称  jeep-大切诺基-2004
	float fConfdence;              //车辆类型置信度
	int iVehicleSubModel;          //车型名称索引
	unsigned char*pCaptureImage; //视频模式识别下抓拍的图片数据，必须调用时开辟，图片识别模式下不必开辟
	                             //开辟大小为输入图片宽高乘积的3倍,且视频模式下每次只输出一个识别结果
	                             //(注：此抓拍结果图的存储顺若输入图像是YUV格式，则为RGBRGB，否则与输如图像顺序相同)
	int iImageWidth;             //视频模式下抓拍图像宽度
	int iImageHeight;            //视频模式下抓拍图像高度

}ITS_Vehicle_Result_Single;


typedef struct
{
	ITS_Vehicle_Result_Single tempResult[MAX_VEHICLE_NUM];
	int iNum;
}ITS_Vehicle_Result;


typedef struct
{
	int iSceneMode;   //场景模式：   0---顶装       1----侧装    2---顶装路径   3---停车场   4----公安
	int iModelMode;   //读取模型方式：0---原始模型   1---修改模型
}ITS_Rec_Param;



//利用车牌识别，车辆检测进行车型识别
VTR_API int  ITS_ThreadInit(char* fullKeyPath);  

VTR_API int  ITS_GetThreadNum();  //获得加密锁支持的线程数目

VTR_API void * ITS_VehicleRecInit(char* modePath, int& iInitFlag, ITS_Rec_Param its_param);//modePath为model文件夹的路径名

VTR_API int ITS_VehicleTypeRec_new_1(void* pInstance,V_Image* pVImage,S_Rect* roi,ITS_Vehicle_Result* pResult, char plateResult[64]);


VTR_API int ITS_VehicleTypeRec(void* pInstance,V_Image* pVImage,S_Rect* roi,ITS_Vehicle_Result* pResult);

VTR_API void ITS_VehicleRecRelease(void* pInstance);
VTR_API void* Vehicle_Type_Classification_Init(char* modePath,int cardnum);


#endif
