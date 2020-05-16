#include <iostream>

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
//如下是opencv的库
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>//opencv的函数要在库函数前
using namespace std;
using namespace cv;
extern"C"
{
#include "VehicleType.h"
}
#define BFT_BITMAP 0x4d42   /* 'BMP' */
#define BI_RGB        0L
#define BI_RLE8       1L
#define BI_RLE4       2L
#define BI_BITFIELDS  3L
#define NUM_THREADS  4   //batch如果很大的话，这里线程就不能太大
int flag_start=0;//定义一个启动标志
// ¶ÁÈ¡bmpžñÊœÎÄŒþ£¬²¢×ª»»³É»Ò¶ÈÍŒ£¬Ä¿Ç°Ö»Ö§³Ö²ÊÉ«ºÍ»Ò¶ÈÍŒÏñ
int ReadBmp(char * FileName, unsigned char * pImg, int *width, int *height)
{
	FILE *hFile;
	unsigned short bfType,biBitCount;
	unsigned int biSize,dwCompression, biClrUsed, bfOffBits;
	int biWidth, biHeight;
	unsigned int dwEffWidth;
	int i;

	hFile=fopen(FileName,"rb");
	if (hFile == NULL) return 0;

	fread(&bfType,2,1,hFile);
	if (bfType != BFT_BITMAP) { //do we have a RC HEADER?
        printf("Not bmp File!\n");
		return 0;
    }
	fseek(hFile,10,SEEK_SET);
	fread(&bfOffBits,4,1,hFile);
	fread(&biSize,4,1,hFile);
	if (biSize!=40) {
		printf("Not common BITMAPINFOHEADER, BITMAPCOREHEADER=12!\n");
		return 0;
	}
	fread(&biWidth,4,1,hFile);
	fread(&biHeight,4,1,hFile);
	fseek(hFile,2,SEEK_CUR); // Ìø¹ý biPlanes
	fread(&biBitCount,2,1,hFile);
	fread(&dwCompression,4,1,hFile);
	fseek(hFile,12,SEEK_CUR); // Ìø¹ý biPlanes
	fread(&biClrUsed,4,1,hFile);

	if (dwCompression!=BI_RGB) {
		printf("Not supported Compression!\n");
		fclose(hFile);
		return 0;
	}

	if (biBitCount!=24) {
		printf("only support 24bit color!\n");
		fclose(hFile);
		return 0;
	}

//	if (biClrUsed!=0) {
//		printf("Palette not supported, Colors=%d\n",biClrUsed);
//		return false;
//	}

/*
	pImg= (unsigned char *) malloc (biHeight*biWidth*3);

	if (pImg==NULL) {
		printf("no memory when alloc image\n");
		fclose(hFile);
		return false;
	}
*/
    dwEffWidth = ((((biBitCount * biWidth) + 31) / 32) * 4);

	*width = biWidth;
	*height= biHeight;

	printf("%d,%d\n",biWidth,biHeight);

	// ¶šÎ»ÍŒÏñÊýŸÝ
	fseek(hFile,bfOffBits,SEEK_SET);
	for(i=0;i<biHeight;i++)
	{
		fread(pImg+(biHeight-1-i)*biWidth*3, dwEffWidth,1,hFile); // read in the pixels
	}

	fclose(hFile);
	return 1;
}
void *mythread(void *threadid)
{
    int tid = *((int*)threadid);
    
    struct  timeval  start1;
    struct  timeval  end1;
    unsigned long timer;
    char modePath[1024] = {0};
    snprintf(modePath,sizeof(modePath), "./model");
    int hhh=tid%4;
    while(flag_start!=tid);
   
    void* pInstance = Type_VehicleInit(modePath,hhh);
	/////////////////////////////////////////////
    flag_start++;   
    while(flag_start<NUM_THREADS);  
	printf("%d  ready\n",tid);
	char imagePath[100] = {0};
	int j=0;
	snprintf(imagePath,sizeof(imagePath), "image/%d.bmp",j);
	cv::Mat Srcimg=cv::imread(imagePath,IMREAD_COLOR);
	cv::Rect  dongres;//这里是将图片外扩了一定像素，实际上这样不够
	dongres.x=1123;
	dongres.y=1299;
	dongres.width=1400-1123;
	dongres.height=1721-1299;
	cv::Mat Srcimg_smallcar=Srcimg(dongres);
	while(1)
	{
		std::vector<Type_Vehicle_Result>  temptemp;//存放结果
		std::vector< std::vector<V_Image*> > pSRCImage;//存放图片中车辆小图
		std::vector<ITS_Vehicle_Result_Detect> headandtailcpuDetect;//存放大图中车辆位置
		temptemp.clear();
		pSRCImage.clear();
		headandtailcpuDetect.clear();
		for(int x=0;x<16;x++)
		{
			std::vector<V_Image*>  imagetemp;
			ITS_Vehicle_Result_Detect  CARTEMP;
			CARTEMP.CarNum=16;
			for(int xx=0;xx<16;xx++)
			{
				V_Image* pImage = (V_Image*)malloc(sizeof(V_Image)* 1);
				pImage->eType = E_BGR;
				pImage->iImageHeight = Srcimg_smallcar.rows;
				pImage->iImageWidth  = Srcimg_smallcar.cols;
				pImage->pImageData = (unsigned char*)malloc(sizeof(unsigned char)* pImage->iImageHeight * pImage->iImageWidth * 3);//
				int iIndex = 0;
				for (int i = 0; i < pImage->iImageHeight; i++)
				{
					unsigned char* p = (unsigned char*)(Srcimg_smallcar.data + Srcimg_smallcar.step * i);
					for (int j = 0; j < pImage->iImageWidth; j++)
					{
						pImage->pImageData[iIndex++] = p[j * 3 + 0];
						pImage->pImageData[iIndex++] = p[j * 3 + 1];
						pImage->pImageData[iIndex++] = p[j * 3 + 2];
					}
				}
				imagetemp.push_back(pImage);
				////////////////
				CARTEMP.iLeft[xx]=0;;
				CARTEMP.iTop[xx]=0;;
				CARTEMP.iRight[xx]=1400-1123;
				CARTEMP.iBottom[xx]=1721-1299;
			}
			pSRCImage.push_back(imagetemp);
			headandtailcpuDetect.push_back(CARTEMP);
		}

		struct timeval tv_begin, tv_end;
		unsigned long timer;
		gettimeofday(&tv_begin, NULL);

        Type_Vehicle_FromGPU(pInstance,pSRCImage,headandtailcpuDetect);
		gettimeofday(&tv_end, NULL);
		timer = 1000000 * (tv_end.tv_sec-tv_begin.tv_sec)+ tv_end.tv_usec-tv_begin.tv_usec;
		printf("%d Type_Vehicle_FromGPU:timer = %ld ms/pic \n",tid,timer/(1000*16*16));//
		for(int x=0;x<pSRCImage.size();x++)
		{
			for(int xx=0;xx<pSRCImage[x].size();xx++){
				if(pSRCImage[x][xx]!=NULL)
				{
					free(pSRCImage[x][xx]->pImageData);
					free(pSRCImage[x][xx]);
				}
			}
			
		}
		
	}//while循环
	Type_VehicleRecRelease(pInstance);
}

int main()
{
     pthread_t threads[NUM_THREADS];
    int indexes[NUM_THREADS];// 用数组来保存i的值
    for(int i=0; i < NUM_THREADS; i++ )
    {      
		indexes[i] = i; //先保存i的值
		// 传入的时候必须强制转换为void* 类型，即无类型指针        
		int rc = pthread_create(&threads[i], NULL,mythread, (void *)&(indexes[i]));
		if (rc)
		{
			cout << "Error:无法创建线程," << rc << endl;
			exit(-1);
		}
    }
    for(int i=0; i < NUM_THREADS; i++ ){      
		pthread_join(threads[i],NULL);//cout << "right:创建线程," << i << endl;
		
    }
       
     

    
    return 0;
}
