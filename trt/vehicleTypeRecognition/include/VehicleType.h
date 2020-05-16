#ifndef _VEHICLETYPE_H_
#define _VEHICLETYPE_H_
#include "inc.h"
#include "AllModuleInclude.h"


#include "opencv2/core/core.hpp"

#define VTR_API extern "C" __attribute__((visibility("default"))) 
using namespace std;
typedef struct
{	
	char tempVehicleType[1024];    //��������--������  jeep-����ŵ��-2004
	float fConfdence;              //�����������Ŷ�
	int iVehicleSubModel;          //������������		
}Type_Vehicle_Result_Single;//һ����


typedef struct
{
	Type_Vehicle_Result_Single tempResult[MAX_CAR_NUM]; //һ�Ŵ�ͼ��16����
	int iNum;
}Type_Vehicle_Result;//һ��ͼƬ

VTR_API std::vector<Type_Vehicle_Result> Type_Vehicle_FromGPU(void *pInstance,
                                                              std::vector<std::vector<V_Image*> > pImage,
                                                              std::vector<ITS_Vehicle_Result_Detect> headandtailcpuDetect);
															  
VTR_API Type_Vehicle_Result_Single Type_VehicleDetect(void* pInstance,V_Image* pImage,S_Rect* roi);
VTR_API void Type_VehicleRecRelease(void* pInstance);
VTR_API void* Type_VehicleInit(char* modePath,int cardnum);


#endif
