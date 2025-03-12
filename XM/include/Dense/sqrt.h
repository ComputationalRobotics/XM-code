#ifndef DENSE_SQRT_H
#define DENSE_SQRT_H

#include <cuda_runtime.h>
#include <cmath>
#include <Utils/memory.h>

template <typename T>
__global__ void sqrtKernel(T* d_out, T* d_in, size_l size) {
    size_l idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_out[idx] = sqrt(d_in[idx]);
    }
}

// take deviceDnTen as input
template <typename T>
void SqrtDn(DeviceDnTen<T>& out, DeviceDnTen<T>& in) {
    int blockSize = 1024;
    int numBlocks = (out.total_size + blockSize - 1) / blockSize;
    sqrtKernel<<<numBlocks, blockSize>>>(out.vals, in.vals, out.total_size);
}

#endif // DENSE_SQRT_H