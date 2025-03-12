#ifndef DENSE_SCALAR_H
#define DENSE_SCALAR_H

#include <iostream>
#include <Utils/memory.h>

// Your code here
// scalr multi matrix kernal
template <typename T>
__global__ void ScaMatkernel(T *A, T scalar, size_l n) {
    size_l idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] *= scalar;
    }
}

// function for dense tensor
template <typename T>
inline void ScaMat(DeviceDnTen<T>& mat, T scalar) {
    // allocate memory for the result
    ScaMatkernel<<<(mat.total_size + 1023) / 1024, 1024>>>(mat.vals, scalar, mat.total_size);

}


#endif // DENSE_SCALAR_H