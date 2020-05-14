#ifndef _ITS_AllMODULEINCLUDE_H_
#define _ITS_AllMODULEINCLUDE_H_
#define MAX_CAR_NUM 16
typedef struct
{
    int CarNum;
    float fConfdence[MAX_CAR_NUM];
    int tempVehicleType[MAX_CAR_NUM];
    int iLeft[MAX_CAR_NUM];
    int iTop[MAX_CAR_NUM];
    int iRight[MAX_CAR_NUM];
    int iBottom[MAX_CAR_NUM];
} ITS_Vehicle_Result_Detect;// single pic

#endif
