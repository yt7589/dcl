#ifndef _ITS_All_MODULE_INCLUDE_H_
#define _ITS_All_MODULE_INCLUDE_H_
#define MAX_CAR_NUM 20
typedef struct
{
    int CarNum;
    float fConfdence[MAX_CAR_NUM];
    int tempVehicleType[MAX_CAR_NUM];
    int iLeft[MAX_CAR_NUM];
    int iTop[MAX_CAR_NUM];
    int iRight[MAX_CAR_NUM];
    int iBottom[MAX_CAR_NUM];
} ITS_Vehicle_Result_Detect;

#endif