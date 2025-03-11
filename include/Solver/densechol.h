#ifndef DENSECHOL_H
#define DENSECHOL_H

#include <iostream>
#include <vector>
#include <Utils/memory.h>

// chol
template <typename T>
inline void DnChol(DeviceSolverDnHandle& cusloverdn_handle, DeviceDnTen<T>& A){
    cusolverDnParams_t params = NULL;
    size_s M = A.dimensions[0];
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace */

    // upper is the same as MATLAB DEFAULT
    // TODO: traits<T>::cuda_data_type is same as what I write in concept.h
    // allocate workspace
    CHECK_CUSOLVER(cusolverDnXpotrf_bufferSize(
        cusloverdn_handle.cusolver_dn_handle, params, CUBLAS_FILL_MODE_UPPER, M, CudaTypeMapper<T>::value, A.vals, M,
        CudaTypeMapper<T>::value, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    // Cholesky factorization
    int *d_info = nullptr;    /* error info */
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
    CHECK_CUSOLVER(cusolverDnXpotrf(
            cusloverdn_handle.cusolver_dn_handle, params, CUBLAS_FILL_MODE_UPPER, M, CudaTypeMapper<T>::value, A.vals, M,
            CudaTypeMapper<T>::value, d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));

    // check error
    int info = 0;
    CHECK_CUDA(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
    }
    else{
         std::printf("Success in DN chol! \n");
    }
    CHECK_CUDA(cudaDeviceSynchronize());

}

#endif // DENSECHOL_H