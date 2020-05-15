//
// Created by yoloCao on 20-2-13.
//

#ifndef NVHTRESIZE_CVCROP_H
#define NVHTRESIZE_CVCROP_H

#include <vector>
#include "AllModuleInclude.h"


int  nvHTCropAndReizeLaunch(float* &cropImages,
                  std::vector<unsigned char *> &cudaSrc,
                  std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                  ITS_Vehicle_Result_Detect *tempCudaDet,
                  std::vector<int> & srcWidth,std::vector<int> & srcHeight,
                  std::vector<float > & mean,std::vector<float > & std,
                  int batchSize=32,int cropW=416,int cropH=416);
ITS_Vehicle_Result_Detect* initTempCudaDet(int cardNum,int oriBatchSize);
float * initCropAndResizeImages(int cardNum,int batchSize,int maxDetNum,
        int maxOutWidth,int maxOutHeight);


#endif //NVHTRESIZE_CVCROP_H
