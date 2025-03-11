#ifndef MATMUL_H
#define MATMUL_H

#include <Utils/check.h>
#include <Utils/memory.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// matrix vector multiplication
template <typename T>
inline void DnMatDnVec(
    DeviceBlasHandle& cublas_H, 
    DeviceDnTen<T>& vecb, const DeviceDnTen<T>& matA, const DeviceDnTen<T>& vecx, const cublasOperation_t trans = CUBLAS_OP_N
) {
    // check the dimension if we define SAFE
    #ifdef SAFE_MODE
        if(matA.num_dims!=2){
            std::cout<<"the dimension of A is wrong! A dims:  "<< matA.num_dims <<std::endl;
            return;
        }
        else if (vecx.num_dims!=1)
        {
            std::cout<<"the dimension of x is wrong! x dims:  "<< vecx.num_dims <<std::endl;
            return;
        }
        else if(vecb.num_dims!=1)
        {
            std::cout<<"the dimension of b is wrong! b dims:  "<< vecb.num_dims <<std::endl;
            return;
        }
    #endif

    const double alpha = 1.0;
    const double beta = 0.0;
    //TODO: this all convert to double, should change
    CHECK_CUBLAS(cublasDgemv(cublas_H.cublas_handle, trans, matA.dimensions[0], matA.dimensions[1], &alpha, matA.vals, matA.dimensions[0], vecx.vals, 1, &beta, vecb.vals, 1) );
    return;
}

// matrix matrix multiplication
template <typename T>
inline void DnMatDnMat(
    DeviceBlasHandle& cublas_H, 
    DeviceDnTen<T>& matC, const DeviceDnTen<T>& matA, const DeviceDnTen<T>& matB,
    const cublasOperation_t transa = CUBLAS_OP_N, const cublasOperation_t transb = CUBLAS_OP_N,
    const double alpha = 1.0, const double beta = 0.0
) {
    // check the dimension if we define SAFE
    #ifdef SAFE_MODE
        if(matA.num_dims!=2){
            std::cout<<"the dimension of A is wrong! A dims:  "<< matA.num_dims <<std::endl;
            return;
        }
        else if (matB.num_dims!=2)
        {
            std::cout<<"the dimension of B is wrong! B dims:  "<< matB.num_dims <<std::endl;
            return;
        }
        else if(matC.num_dims!=2)
        {
            std::cout<<"the dimension of C is wrong! C dims:  "<< matC.num_dims <<std::endl;
            return;
        }
    #endif

    size_l m = matA.dimensions[0];
    size_l n = matB.dimensions[1];
    size_l k = matA.dimensions[1];
    //TODO: this all convert to double, should change
    if(transb == CUBLAS_OP_T){
        n = matB.dimensions[0];         
    }
    if(transa == CUBLAS_OP_T){
        m = matA.dimensions[1];   
        k = matA.dimensions[0];      
    }
    CHECK_CUBLAS(cublasDgemm(cublas_H.cublas_handle,
                           transa, transb,
                           m, n, k,
                           &alpha,
                           matA.vals, matA.dimensions[0],
                           matB.vals, matB.dimensions[0],
                           &beta,
                           matC.vals, matC.dimensions[0]));        

    return;
}

// dnmat1[i] <-- op(dnmat2[i]) * op(dnmat3[i]), i = 1 ... batch_size
// op() is either identity or transpose
template <typename T>
inline void DnMatDnMatBatch(
    DeviceBlasHandle& cublas_H, 
    DeviceDnTen<T>& matC, const DeviceDnTen<T>& matA, const DeviceDnTen<T>& matB,
    const size_s m,  const size_s k,  const size_s n, const size_s num_mat,
    const cublasOperation_t transa = CUBLAS_OP_N, const cublasOperation_t transb = CUBLAS_OP_N,
    const double alpha = 1.0, const double beta = 0.0
) {
    const size_l strideA = m*k;
    const size_l strideB = k*n;
    const size_l strideC = m*n;
    CHECK_CUBLAS(cublasDgemmStridedBatched(
        cublas_H.cublas_handle, transa, transb,
        m,n,k,
        &alpha, matA.vals, matA.dimensions[0], strideA, matB.vals, matB.dimensions[0], strideB,
        &beta, matC.vals, matC.dimensions[0], strideC,
        num_mat
    ) );
    return;
}

#endif