//
// Created by yoloCao on 20-2-13.
//

#ifndef NVRESIZE_CVCROP_NOVIO_H
#define NVRESIZE_CVCROP_NOVIO_H

#include <vector>
#include "CarFeature.h"

int  nvCropAndResizeAndNormLaunch(float *cropImages,
                                  const std::vector<unsigned char *> &cudaSrc,
                                  ITS_Vehicle_Result_Detect *tempCudaDet,
                                  std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                                  const std::vector<int> &srcWidth, const std::vector<int> &srcHeight,
                                  int batchSize, int cropW, int cropH,float means[3],float stds[3]);

int nvCropAndResizeAndNormLaunch(float *cropImages,
                                  const std::vector<unsigned char *> &cudaSrc,
                                  ITS_Vehicle_Result_Detect *tempCudaDet,
                                  ITS_Vehicle_Result_Detect *tempCudaPDet,
                                  std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                                  const std::vector<int> &srcWidth, const std::vector<int> &srcHeight,
                                  int batchSize, int cropW, int cropH,float means[3],float stds[3]);


ITS_Vehicle_Result_Detect *initTempCudaDet(int maxDetNum);

void freePointer(void *p);

#endif //NVRESIZE_CVCROP_NOVIO_H
