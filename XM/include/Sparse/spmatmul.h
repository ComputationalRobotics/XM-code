#ifndef SPMATMUL_H
#define SPMATMUL_H

#include <iostream>
#include <vector>
#include <cassert>
#include <Utils/memory.h>
#include <cusparse.h>
template <typename T>
inline size_t SpMV_get_buffersize_cusparse(
    DeviceSparseHandle& cusparse_H, 
    const DeviceSpMatCSR<T>& A, const DeviceDnTen<T>& x, DeviceDnTen<T>& y, 
    const double alpha, const double beta
) {
    size_t buffer_size = 0;
    // Provides deterministic (bit-wise) results for each run only for CUSPARSE_SPMV_COO_ALG2 and
    // CUSPARSE_SPMV_CSR_ALG2 algorithms, and opA == CUSPARSE_OPERATION_NON_TRANSPOSE
    // so here default is non-transpose
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        cusparse_H.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.cusparse_descr, x.cusparse_descr, 
        &beta, y.cusparse_descr,
        CudaTypeMapper<T>::value, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size
    ) );
    // in doc it said both CSC/CSR can choose CUSPARSE_SPMV_CSR_ALG1
    return buffer_size;
}


// CSC and CSR is the same code
template <typename T>
inline void SpMatDnVec(
    DeviceSparseHandle& cusparse_H, 
    const DeviceSpMatCSR<T>& A, const DeviceDnTen<T>& x, DeviceDnTen<T>& y, 
    const double alpha, const double beta,
    void* buffer
) {
    CHECK_CUSPARSE( cusparseSpMV(
        cusparse_H.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.cusparse_descr, x.cusparse_descr, 
        &beta, y.cusparse_descr,
        CudaTypeMapper<T>::value, CUSPARSE_SPMV_ALG_DEFAULT, buffer
    ) );
    return;
}

// //spGEMM for three CSR matrix
// template <typename T>
// inline void SpMatSpMat(
//     DeviceSparseHandle& cusparse_H, 
//     const DeviceSpMatCSR<T>& A, const DeviceDnTen<T>& x, DeviceDnTen<T>& y, 
//     const double alpha, const double beta,
//     void* buffer
// ) {
//     CHECK_CUSPARSE( cusparseSpMV(
//         cusparse_H.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//         &alpha, A.cusparse_descr, x.cusparse_descr, 
//         &beta, y.cusparse_descr,
//         CudaTypeMapper<T>::value, CUSPARSE_SPMV_ALG_DEFAULT, buffer
//     ) );
//     return;
// }


#endif // SPMATMUL_H