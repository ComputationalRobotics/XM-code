
#ifndef MATDIAGMUL_H
#define MATDIAGMUL_H

#include <vector>
#include <Utils/memory.h>
#include <iostream>

// dnmat1[i] <-- dnmat2[i] * a[i], i = 1 ... batch_size
// where dnmat1[i] and dnmat2[i] is of size mat_size * mat_size,
// a[i] is of size 1
// all the matrices are stored in colomn-major order,which means the address in one matric is not continuous
/**
 * @brief Kernel function to multiply each column of a dense matrix by a scalar value in a batched manner.
 *
 * This kernel performs element-wise multiplication of each column of the input dense matrix `dnmat2_vals`
 * by the corresponding scalar value from `dnscaler_vals`, and stores the result in `dnmat1_vals`.
 * The operation is performed for a batch of matrices.
 *
 * @tparam T The data type of the matrix and scalar values.
 * @param dnmat1_vals Pointer to the output dense matrix values.
 * @param dnmat2_vals Pointer to the input dense matrix values.
 * @param dnscaler_vals Pointer to the scalar values for each column.
 * @param mat_size The size of the matrix (number of rows/columns).
 * @param n The number of matrices in the batch.
 */

template <typename T>
__global__ void dnmat_mul_scaler_colomn_batch_kernel(
    T* dnmat1_vals, T* dnmat2_vals, T* dnscaler_vals, 
    size_s mat_size_r, size_s mat_size_c, size_s n
) {
    size_l total_len = n * mat_size_r * mat_size_c;
    size_s idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_s i;
    if (idx < total_len) {
        i = (idx % (mat_size_r * n))/mat_size_r;
        dnmat1_vals[idx] = dnmat2_vals[idx] * dnscaler_vals[i];
    }
    return;
}

// dnmat1[i] <-- dnmat2[i] * diag(dnvec[i]), i = 1 ... batch_size
// where dnmat1[i] and dnmat2[i] is of size mat_size * mat_size,
// dnvec[i] is of size mat_size
template <typename T>
void dnmat_mul_spdiag_batch(
    DeviceDnTen<T>& dnmat1, const DeviceDnTen<T>& dnmat2, const DeviceDnTen<T>& dnvec,
    const int mat_size_r, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
) {
    int num_block = (dnmat1.total_size + block_size - 1) / block_size;
    dnmat_mul_scaler_colomn_batch_kernel<<<num_block, block_size, 0, stream>>>(
        dnmat1.vals, dnmat2.vals, dnvec.vals,
        mat_size_r, dnmat1.dimensions[1], dnvec.dimensions[0] 
    );
    return;
}

// batched Ddot: for a 3n * o matrix dnmat1, a 3n * o matrix dnmat2, and a n * 1 matrix dnvec
// do 3*o dot products for each n, and store the result in dnmat1
template <typename T>
__global__ void dnmat_Ddot_colomn_batch_kernel(
    T* dnresult_vals, T* dnmat1_vals, T* dnmat2_vals,  
    size_s mat_size_r, size_s mat_size_c, size_s n
) {
    size_s idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n && idx > 0) {
        dnresult_vals[idx-1] = 0;
        for( int i = 0; i < mat_size_c; i++) {
            int k = i * mat_size_r * n + idx * mat_size_r;
            for( int j = 0; j < mat_size_r; j++) {
                dnresult_vals[idx-1] += dnmat2_vals[k + j] * dnmat1_vals[k + j];
            }
        }
    }
    return;
}

template <typename T>
void dnmat_Ddot_colomn_batch(
    const DeviceDnTen<T>& dnvec, DeviceDnTen<T>& dnmat1, const DeviceDnTen<T>& dnmat2, 
    const int mat_size_r, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
) {
    int num_block = (dnvec.total_size + block_size - 1) / block_size;
    dnmat_Ddot_colomn_batch_kernel<<<num_block, block_size, 0, stream>>>(
        dnvec.vals, dnmat1.vals, dnmat2.vals, 
        mat_size_r, dnmat1.dimensions[1], dnvec.dimensions[0]+1
    );
    return;
}

#endif