//
// Created by yoloCao on 20-2-13.
//

#ifndef __NV_CROP_AND_RESIZE_NOVIO_H__
#define __NV_CROP_AND_RESIZE_NOVIO_H__

#include <vector>
#include "CarFeature.h"

void nvCropAndResizeAndNormLaunch(float *cropImages,
                                  const std::vector<unsigned char *> &cudaSrc,
                                  ITS_Vehicle_Result_Detect *tempCudaDet,
std::vector<ITS_Vehicle_Result_Detect> &cpuDet,
                                  const std::vector<int> &srcWidth, const std::vector<int> &srcHeight,
                                  int batchSize, int cropW, int cropH,float means[3],float stds[3]);

ITS_Vehicle_Result_Detect *initTempCudaDet(int maxDetNum);

void freePointer(void *p);

#endif // __NV_CROP_AND_RESIZE_NOVIO_H__
