#include <stdlib.h>
#include <stdio.h>

#include <memory.h>

#include <sys/time.h>
#include <time.h>
#include <string>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iconv.h>
#include <pthread.h>


#include <algorithm>


#include "VehicleType.h" 
#include "ItsVehicleTypeRec.h"  //车品牌头文件
using namespace std;
typedef struct
{
	void* pVtr;//品牌的句柄
}Type_Data;
void* Type_VehicleInit(char* modePath,int cardnum)
{
	Type_Data *pData = new(Type_Data);
	pData->pVtr = Vehicle_Type_Classification_Init(modePath, cardnum);
	return pData;
}
Type_Vehicle_Result_Single Type_VehicleDetect(void* pInstance,V_Image* pImage,S_Rect* roi)
{
	Type_Vehicle_Result_Single pResult;
	memset(&pResult,0,sizeof(Type_Vehicle_Result_Single));

	if(0==pImage->iImageHeight || pImage->iImageWidth==0)
		return pResult;
	if(NULL==pInstance)
		return pResult;
	Type_Data *pData = (Type_Data*) pInstance;
	if(NULL==pData)
		return pResult;
	//定义一个临时结构体，存放车辆的品牌和可信度
	ITS_Vehicle_Result sResult;
	memset(&sResult,0,sizeof(ITS_Vehicle_Result));
	int iVehicleNum=0;
	iVehicleNum = ITS_VehicleTypeRec(pData->pVtr, pImage, roi, &sResult);
	if(iVehicleNum){
		pResult.fConfdence= sResult.tempResult[0].fConfdence;//品牌型号的置信度
		pResult.iVehicleSubModel= sResult.tempResult[0].iVehicleSubModel;//品牌型号的索引
		strcpy(pResult.tempVehicleType,sResult.tempResult[0].tempVehicleType);//品牌型号年款
	}	
	return pResult;
}
VTR_API std::vector<Type_Vehicle_Result>Type_Vehicle_FromGPU(void *pInstance,
															std::vector< std::vector<V_Image*> > pImage,
															std::vector<ITS_Vehicle_Result_Detect> headandtailcpuDetect)
{//在检测函数Type_VehicleDetect中进行类型转换，在此不需要转换
	std::vector<Type_Vehicle_Result>  temp;
	temp.clear();//
	if(NULL!=pInstance)
    {
		//到这里之后，有可能送过来的图片都没有解码成功，因此这里需要判断一下,同时要把图片进行车辆检测分类，在这里调用车辆检测函数
		for(int xw=0;xw<headandtailcpuDetect.size();xw++)//针对每一张图片
		{
			Type_Vehicle_Result sResult;// = { 0 };
			memset(&sResult,0,sizeof(Type_Vehicle_Result));
			sResult.iNum=headandtailcpuDetect[xw].CarNum;//当前图片上车辆数量
			for(int y=0;y<headandtailcpuDetect[xw].CarNum;y++)//针对每一个车
			{
				S_Rect roi;//这个是相对于这幅图像的区域，是个相对值，之前传的都是相对大图而言的就有错误了
				roi.iLeft =0;
				roi.iTop = 0;
				roi.iRight =headandtailcpuDetect[xw].iRight[y]-headandtailcpuDetect[xw].iLeft[y];
				roi.iBottom =headandtailcpuDetect[xw].iBottom[y]-headandtailcpuDetect[xw].iTop[y];
				try  //尝试做车辆检测
				{
					sResult.tempResult[y]=Type_VehicleDetect(pInstance,pImage[xw][y],&roi);
				}
				catch (std::exception &ex)
				{
					std::cout << "Detect_CarNum_Gpu exception caught: " << ex.what() << std::endl;
				}
				
			}
			temp.push_back(sResult);//将当前图片中车辆的品牌存起来
		}	
    }
    return temp;//这里额外注意
}
void Type_VehicleRecRelease(void* pInstance)
{
	Type_Data *pData = (Type_Data*) pInstance;
	if(pData->pVtr)
	{
		ITS_VehicleRecRelease(pData->pVtr);
		pData->pVtr=NULL;
	}
	free(pData);
}
