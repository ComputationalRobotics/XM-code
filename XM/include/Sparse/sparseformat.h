#ifndef SPARSEFORMAT_H
#define SPARSEFORMAT_H

#include <iostream>
#include <Utils/memory.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_ptr.h>



// COOsort
template <typename T>
inline void COOsort(
    int* row_ids, int* col_ids, T* vals, const size_l nnz
) {
    // Use zip_iterator to create tuples (row, col, val) for sorting
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(row_ids), thrust::device_pointer_cast(col_ids), thrust::device_pointer_cast(vals)));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(row_ids + nnz), thrust::device_pointer_cast(col_ids + nnz), thrust::device_pointer_cast(vals + nnz)));

    // Sort by row first, then by column for identical rows
    thrust::sort(begin, end, thrust::less<thrust::tuple<int, int, T>>());
    return;
}

// convert CSC format in cusparse to CSR format in cusparse
// this routine needs additional memories on device
// here we assume mat_csr's memory has already been allocated
template <typename T>
inline size_t CSC2CSR_get_buffersize(
    DeviceSparseHandle& cusparse_H,
    const DeviceSpMatCSC<T>& mat_csc, DeviceSpMatCSR<T>& mat_csr
) {
    size_t buffer_size;
    CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(
        cusparse_H.cusparse_handle, 
        mat_csc.col_size, mat_csc.row_size, mat_csc.nnz,
        mat_csc.vals, mat_csc.col_ptrs, mat_csc.row_ids, 
        mat_csr.vals, mat_csr.row_ptrs, mat_csr.col_ids, 
        CudaTypeMapper<T>::value, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, 
        &buffer_size
    ) );
    return buffer_size;
}

// suppose buffer already been allocated
template <typename T>
inline void CSC2CSR(
    DeviceSparseHandle& cusparse_H,
    const DeviceSpMatCSC<T>& mat_csc, DeviceSpMatCSR<T>& mat_csr
) {
    size_t buffer_size = CSC2CSR_get_buffersize(cusparse_H, mat_csc, mat_csr);
    void* buffer;
    CHECK_CUDA( cudaMalloc(&buffer, buffer_size) );
    CHECK_CUSPARSE( cusparseCsr2cscEx2(
        cusparse_H.cusparse_handle, 
        mat_csc.col_size, mat_csc.row_size, mat_csc.nnz,
        mat_csc.vals, mat_csc.col_ptrs, mat_csc.row_ids, 
        mat_csr.vals, mat_csr.row_ptrs, mat_csr.col_ids, 
        CudaTypeMapper<T>::value, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
        buffer
    ) );
    return;
}

template <typename T>
inline size_t CSR2CSC_get_buffersize(
    DeviceSparseHandle& cusparse_H,
    const DeviceSpMatCSR<T>& mat_csr, DeviceSpMatCSC<T>& mat_csc
) {
    size_t buffer_size;
    CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(
        cusparse_H.cusparse_handle, 
        mat_csr.row_size, mat_csr.col_size, mat_csr.nnz,
        mat_csr.vals, mat_csr.row_ptrs, mat_csr.col_ids, 
        mat_csc.vals, mat_csc.col_ptrs, mat_csc.row_ids, 
        CudaTypeMapper<T>::value, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, 
        &buffer_size
    ) );
    return buffer_size;
}

// suppose buffer already been allocated
template <typename T>
inline void CSR2CSC(
    DeviceSparseHandle& cusparse_H,
    const DeviceSpMatCSR<T>& mat_csr, DeviceSpMatCSC<T>& mat_csc
) {
    size_t buffer_size = CSR2CSC_get_buffersize(cusparse_H, mat_csr, mat_csc);
    void* buffer;
    CHECK_CUDA( cudaMalloc(&buffer, buffer_size) );
    CHECK_CUSPARSE( cusparseCsr2cscEx2(
        cusparse_H.cusparse_handle, 
        mat_csr.row_size, mat_csr.col_size, mat_csr.nnz,
        mat_csr.vals, mat_csr.row_ptrs, mat_csr.col_ids, 
        mat_csc.vals, mat_csc.col_ptrs, mat_csc.row_ids, 
        CudaTypeMapper<T>::value, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, 
        buffer
    ) );
    return;
}



#endif // SPARSEFORMAT_H