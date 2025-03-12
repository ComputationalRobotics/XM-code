#ifndef DENSE_EIG_H
#define DENSE_EIG_H

#include <iostream>
#include <Utils/memory.h>

// Your code goes here
// single matrix eig parameter
class SingleEigParam {
    public:
        int gpu_id;
        cusolverEigMode_t jobz;
        cublasFillMode_t uplo;
        cusolverDnParams_t param;

        SingleEigParam(const int gpu_id = 0): gpu_id(gpu_id) {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            this->param = NULL;
            CHECK_CUSOLVER( cusolverDnCreateParams(&this->param) );
            // we need eigen vector
            this->jobz = CUSOLVER_EIG_MODE_VECTOR;
            // upper and lower is arbitrary
            this->uplo = CUBLAS_FILL_MODE_LOWER;
        }

        ~SingleEigParam() {
            if (this->param != NULL) {
                CHECK_CUSOLVER( cusolverDnDestroyParams(this->param) );
                this->param = NULL;
            }
        }
};

// get single matrix eig buffer sizes in cusolver
template <typename T>
inline void single_eig_get_buffersize_cusolver(
    DeviceSolverDnHandle& cusolver_H, SingleEigParam& param,
    DeviceDnTen<T>& mat, DeviceDnTen<T>& W, 
    const size_s mat_size,
    size_t* buffer_size, size_t* buffer_size_host,
    const int mat_offset = 0, const int W_offset = 0
) {
    CHECK_CUDA( cudaSetDevice(cusolver_H.gpu_id) );
    CHECK_CUSOLVER( cusolverDnXsyevd_bufferSize(
        cusolver_H.cusolver_dn_handle, param.param, param.jobz, param.uplo,
        mat_size, CudaTypeMapper<T>::value, mat.vals + mat_offset,
        mat_size, CudaTypeMapper<T>::value, W.vals + W_offset,
        CudaTypeMapper<T>::value,
        buffer_size, buffer_size_host
    ) );
}

// calculate single matrix eig in cusolver
template <typename T>
inline void single_eig_cusolver(
    DeviceSolverDnHandle& cusolver_H, SingleEigParam& param,
    DeviceDnTen<T>& mat, DeviceDnTen<T>& W, 
    void* buffer, void* buffer_host, int* info,
    const size_s mat_size, const size_t buffer_size, const size_t buffer_size_host,
    const int mat_offset = 0, const int W_offset = 0,
    const int info_offset = 0
) {
    CHECK_CUDA( cudaSetDevice(cusolver_H.gpu_id) );
    CHECK_CUSOLVER( cusolverDnXsyevd(
        cusolver_H.cusolver_dn_handle, param.param, param.jobz, param.uplo,
        mat_size, CudaTypeMapper<T>::value, mat.vals + mat_offset,
        mat_size, CudaTypeMapper<T>::value, W.vals + W_offset,
        CudaTypeMapper<T>::value, 
        buffer, buffer_size,
        buffer_host, buffer_size_host,
        info + info_offset
    ) );
}
#endif // DENSE_EIG_H