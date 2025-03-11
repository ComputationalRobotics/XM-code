#ifndef TRACE_H
#define TRACE_H

#include <Utils/check.h>
#include <Utils/memory.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// diag of a square matrix
template <typename T>
__global__ void DiagKernel(T* mat, T* diag, size_s N) {
    // mat is the input square matrix, and diag is the vector output
   size_l idx = threadIdx.x + blockIdx.x * blockDim.x;
   if(idx < N){
         diag[idx] = mat[idx*N+idx];
   }

}

template <typename T>
__global__ void traceKernel(T* mat, T* result, size_s N) {
    // mat is the input square matrix, and result is a scalar output
    size_l idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        atomicAdd(&result[0], mat[idx*N+idx]);
    }
    
}

template <typename T>
inline void trace(DeviceDnTen<T>& mat, T* result){
    
    // allocate memory for the result
    traceKernel<<<(mat.dimensions[0] + 1023) / 1024, 1024>>>(mat.vals, result, mat.dimensions[0]);
 
}

// diagnal add a matrix
template <typename T>
__global__ void DiagAddKernel(T* mat, T* diag, size_s N) {
    // mat is the input square matrix, and diag is the vector output
   size_l idx = threadIdx.x + blockIdx.x * blockDim.x;
   if(idx < N){
         atomicAdd(&mat[idx*N+idx],diag[idx]);
   }
}

template <typename T>
inline void DiagAdd(DeviceDnTen<T>& mat, DeviceDnTen<T>& diag){
    // allocate memory for the result
    DiagAddKernel<<<(mat.dimensions[0] + 1023) / 1024, 1024>>>(mat.vals, diag.vals, mat.dimensions[0]);
 
}

#endif