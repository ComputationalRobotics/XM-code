#ifndef SPARSE_SUBMATRIX_H
#define SPARSE_SUBMATRIX_H

#include <iostream>
#include <vector>
#include <Utils/memory.h>

// Your code here
// initialize kernal
__global__ void MarkRowKernel(int* row, int* col, size_s start, size_s end, size_l nnz, int* flag){
    size_l i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nnz){
        if(row[i] >= start && row[i] <= end){
            flag[i] = 1;
        }
    }
}

// initialize kernal
__global__ void MarkColKernel(int* row, int* col, size_s start, size_s end, size_l nnz, int* flag){
    size_l i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nnz){
        if(col[i] >= start && col[i] <= end){
            flag[i] = 1;
        }
    }
}

// // take a submatrix of a sparse matrix CSR (only take rows)
// __global__ void SubRowKernel(int* row, int* col, size_s start, size_s end, size_l nnz, int* flag){
//     size_l i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i < nnz){
//         if(col[i] >= start && col[i] <= end){
//             flag[i] = 1;
//         }
//     }
// }

#endif // SPARSE_SUBMATRIX_H