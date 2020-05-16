#ifndef _ITS_VEHICLE_TYPE_REC_H_
#define _ITS_VEHICLE_TYPE_REC_H_
#include "inc.h" 
#define VTR_API extern "C" __attribute__((visibility("default"))) 





typedef struct
{	
	char tempVehicleType[1024];    //��������--������  jeep-����ŵ��-2004
	float fConfdence;              //�����������Ŷ�
	int iVehicleSubModel;          //������������
	unsigned char*pCaptureImage; //��Ƶģʽʶ����ץ�ĵ�ͼƬ���ݣ��������ʱ���٣�ͼƬʶ��ģʽ�²��ؿ���
	                             //���ٴ�СΪ����ͼƬ��߳˻���3��,����Ƶģʽ��ÿ��ֻ���һ��ʶ����
	                             //(ע����ץ�Ľ��ͼ�Ĵ洢˳������ͼ����YUV��ʽ����ΪRGBRGB������������ͼ��˳����ͬ)
	int iImageWidth;             //��Ƶģʽ��ץ��ͼ����
	int iImageHeight;            //��Ƶģʽ��ץ��ͼ��߶�

}ITS_Vehicle_Result_Single;


typedef struct
{
	ITS_Vehicle_Result_Single tempResult[MAX_VEHICLE_NUM];
	int iNum;
}ITS_Vehicle_Result;


typedef struct
{
	int iSceneMode;   //����ģʽ��   0---��װ       1----��װ    2---��װ·��   3---ͣ����   4----����
	int iModelMode;   //��ȡģ�ͷ�ʽ��0---ԭʼģ��   1---�޸�ģ��
}ITS_Rec_Param;



//���ó���ʶ�𣬳��������г���ʶ��
VTR_API int  ITS_ThreadInit(char* fullKeyPath);  

VTR_API int  ITS_GetThreadNum();  //��ü�����֧�ֵ��߳���Ŀ

VTR_API void * ITS_VehicleRecInit(char* modePath, int& iInitFlag, ITS_Rec_Param its_param);//modePathΪmodel�ļ��е�·����

VTR_API int ITS_VehicleTypeRec_new_1(void* pInstance,V_Image* pVImage,S_Rect* roi,ITS_Vehicle_Result* pResult, char plateResult[64]);


VTR_API int ITS_VehicleTypeRec(void* pInstance,V_Image* pVImage,S_Rect* roi,ITS_Vehicle_Result* pResult);

VTR_API void ITS_VehicleRecRelease(void* pInstance);
VTR_API void* Vehicle_Type_Classification_Init(char* modePath,int cardnum);


#endif
