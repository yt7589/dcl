#ifndef _INC_H_
#define _INC_H_

#define  MAX_VEHICLE_NUM  (5)

#define  MAX_FACE_NUM     (10)



#define  OK                 (0)     //³õÊŒ»¯ÕýÈ·

#define  MODE_PATH_ERROR    (-3)    //modeÎÄŒþŽíÎó
typedef enum 
{
	E_RGB,
	E_BGR,
	E_YUV420,  
}V_ImageType;  //图片格式

typedef struct
{
	int       iImageHeight;
	int       iImageWidth;
	V_ImageType eType;
	unsigned char *pImageData;
}V_Image; //图像格式
typedef struct  
{
	int iLeft;
	int iTop;
	int iRight;
	int iBottom;
}S_Rect;
#endif
