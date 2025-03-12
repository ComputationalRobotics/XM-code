#ifndef DENSE_TRANSPOSE_H
#define DENSE_TRANSPOSE_H

#include <Utils/memory.h>
#include <iostream>

template <typename T>
__global__ void transposeKernel(T* out, const T* in, int width, int height) {
    size_l x = blockIdx.x * blockDim.x + threadIdx.x;
    size_l y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}
// transpose a matrix
template <typename T>
void transpose(T* out, const T* in, size_s width, size_s height) {
    dim3 blockDim(32, 32); 
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y); 
    transposeKernel<<<gridDim, blockDim>>>(out, in, width, height);
}

template <typename T>
__global__ void symKernel(T* out, const T* in, int mat_size) {
    size_l x = blockIdx.x * blockDim.x + threadIdx.x;
    size_l y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < mat_size && y < mat_size) {
        out[x * mat_size + y] = (in[x * mat_size + y] + in[y * mat_size + x])*0.5;
    }
}
// transpose a matrix
template <typename T>
void sym(T* out, const T* in, size_s mat_size) {
    dim3 blockDim(32, 32); 
    dim3 gridDim((mat_size + blockDim.x - 1) / blockDim.x, (mat_size + blockDim.y - 1) / blockDim.y); 
    symKernel<<<gridDim, blockDim>>>(out, in, mat_size);
}

template <typename T>
__global__ void symBatchedKernel(T* out, const T* in, size_s mat_size, size_s batch_size) {
    size_l x = blockIdx.x * blockDim.x + threadIdx.x;
    size_l y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < mat_size && y < mat_size * batch_size) {
        size_s bid = y / mat_size;
        size_s y_elem = y % mat_size;
        out[bid * mat_size * mat_size + y_elem * mat_size + x] = (in[bid * mat_size * mat_size + x * mat_size + y_elem] + in[bid * mat_size * mat_size + y_elem * mat_size + x])*0.5;
    }
}
// transpose a matrix
template <typename T>
void symBatched(T* out, const T* in, size_s mat_size, size_s batch_size) {
    dim3 blockDim(32, 32); 
    dim3 gridDim((mat_size + blockDim.x - 1) / blockDim.x, (mat_size * batch_size  + blockDim.y - 1) / blockDim.y); 
    symBatchedKernel<<<gridDim, blockDim>>>(out, in, mat_size, batch_size);
}

#endif // DENSE_TRANSPOSE_H