#ifndef DENSE_MATDIVIDE_H
#define DENSE_MATDIVIDE_H

#include <vector>
#include <Utils/memory.h>
#include <iostream>

template <typename T>
__global__ void dnmat_divide_kernel(
    T* dnmat3_vals, T* dnmat1_vals, T* dnmat2_vals, 
    size_s total_len, const size_s power
) {
    size_l idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total_len) {
        if(power == 1) dnmat3_vals[idx] = dnmat1_vals[idx] / dnmat2_vals[idx];
        else dnmat3_vals[idx] = dnmat1_vals[idx] / pow(dnmat2_vals[idx],power);
    }
    return;
}

// cpu function of Ddot
template <typename T>
void DnMatDnMatDivide(
    DeviceDnTen<T>& dnmat3, DeviceDnTen<T>& dnmat1, const DeviceDnTen<T>& dnmat2, const size_s power,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
) {
    int num_block = (dnmat1.total_size + block_size - 1) / block_size;
    dnmat_divide_kernel<<<num_block, block_size, 0, stream>>>(
        dnmat3.vals, dnmat1.vals, dnmat2.vals, dnmat1.total_size, power
    );
    return;
}

#endif // DENSE_MATDOT_H