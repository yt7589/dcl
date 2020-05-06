//
// Created by novio on 20-2-12.
//
#include "process_input_gpu.h"

struct norm_op {
    float mean, std;

    norm_op(float mean, float std) : mean(mean), std(std) {}

    template<typename _T>
    __device__
    float operator()(_T t) {
        return (static_cast<float>(t) / 255.0f - mean) / std;
    }
};

void gpuMemTransform(cudaStream_t &stream, uint8_t *startPtr, uint8_t *endPtr, float *targetStartPtr, float mean,
                     float stdValue) {
    thrust::transform(
            thrust::cuda::par.on(stream),
            startPtr,
            endPtr,
            targetStartPtr,
            norm_op(mean, stdValue));
}
