#ifndef DENSE_MATDOT_H
#define DENSE_MATDOT_H

#include <vector>
#include <Utils/memory.h>
#include <iostream>

/**
 * @brief Kernel function to perform dot product of two matrices
 *
 * This kernel performs dot product of two matrices `dnmat1_vals` and `dnmat2_vals`
 * The result is stored in `dnmat1_vals`.
 *
 * @tparam T The data type of the matrix values.
 * @param dnmat1_vals Pointer to the output matrix values.
 * @param dnmat2_vals Pointer to the input matrix values.
 * @param mat_size_r The number of rows in the matrices.
 * @param mat_size_c The number of columns in the matrices.
 * @param n The number of matrices in the batch.
 */
template <typename T>
__global__ void dnmat_dot_kernel(
    T* dnmat3_vals, T* dnmat1_vals, T* dnmat2_vals, 
    size_s total_len, const size_s power
) {
    size_l idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total_len) {
        if(power == 1) dnmat3_vals[idx] = dnmat1_vals[idx] * dnmat2_vals[idx];
        else dnmat3_vals[idx] = dnmat1_vals[idx] * pow(dnmat2_vals[idx],power);
    }
    return;
}

// cpu function of Ddot
template <typename T>
void DnMatDnMatDot(
    DeviceDnTen<T>& dnmat3, DeviceDnTen<T>& dnmat1, const DeviceDnTen<T>& dnmat2, const size_s power,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
) {
    int num_block = (dnmat1.total_size + block_size - 1) / block_size;
    dnmat_dot_kernel<<<num_block, block_size, 0, stream>>>(
        dnmat3.vals, dnmat1.vals, dnmat2.vals, dnmat1.total_size, power
    );
    return;
}

#endif // DENSE_MATDOT_H