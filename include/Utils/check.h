#ifndef CUOPT_CHECK_H
#define CUOPT_CHECK_H

#include <cuda_runtime_api.h> 
#include <cusparse.h>  
#include <cublas_v2.h>   
#include <cusolverDn.h>
#include <iostream>


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
do {                                                                                           \
    cudaError_t err_ = (err);                                                                  \
    if (err_ != cudaSuccess) {                                                                 \
        printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
        throw std::runtime_error("CUDA error");                                                \
    }                                                                                          \
} while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
do {                                                                                           \
    cusolverStatus_t err_ = (err);                                                             \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
        printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
        throw std::runtime_error("cusolver error");                                            \
    }                                                                                          \
} while (0)
// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// check Cuda error
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

// check cuBLAS error
#define CHECK_CUBLAS(err)                                                      \
do {                                                                           \
    cublasStatus_t err_ = (err);                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                       \
        printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);   \
    }                                                                          \
} while (0)

// cusolver API error checking
#define CHECK_CUSOLVER(err)                                                    \
do {                                                                           \
    cusolverStatus_t err_ = (err);                                             \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                                     \
        printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);      \
    }                                                                          \
} while (0)

// check cuSPARSE error
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

// capture kernel error
inline void get_kernel_launch_err() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize(); // Ensure all preceding CUDA calls have completed
    return;
}


#endif