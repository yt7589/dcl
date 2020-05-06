//
// Created by novio on 20-2-12.
//

#ifndef __PROCESS_INPUT_GPU_H__
#define __PROCESS_INPUT_GPU_H__

#include <thrust/transform.h>
#include <thrust/system/cuda/detail/par.h>

void gpuMemTransform(cudaStream_t &stream,uint8_t* startPtr,uint8_t* endPtr,float* targetStartPtr,float,float);
#endif // __PROCESS_INPUT_GPU_H__