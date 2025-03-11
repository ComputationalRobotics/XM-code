#ifndef SPARSE_STACK_H
#define SPARSE_STACK_H

#include <cuda_runtime.h>
#include <iostream>

// Function to stack CSR matrices in a column
// This function stacks two CSR matrices in a column
// The output matrix is the concatenation of the two input matrices
// The output matrix is stored in CSR format
// C = [alpha*A; beta*B]
__global__ void stackCSRColumn(const int* row_ptr1, const int* col_ind1, const double* values1, int nnz1,
                               const int* row_ptr2, const int* col_ind2, const double* values2, int nnz2,
                               int* row_ptr_out, int* col_ind_out, double* values_out, int num_rows1, int num_rows2,
                               double alpha, double beta){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nnz1){
        col_ind_out[i] = col_ind1[i];
        values_out[i] = alpha * values1[i];
    }
    if(i < nnz2){
        col_ind_out[i + nnz1] = col_ind2[i];
        values_out[i + nnz1] = beta * values2[i];
    }
    if(i < num_rows1){
        row_ptr_out[i] = row_ptr1[i];
    }
    if(i < num_rows2){
        row_ptr_out[i + num_rows1] = row_ptr2[i] + nnz1;
    }
}

// Function to stack CSR matrices in a row
// This function stacks two CSR matrices in a row
// The output matrix is the concatenation of the two input matrices
// The output matrix is stored in CSR format
// C = [alpha*A, beta*B]

__global__ void stackCSRrow(const int* row_ind1,const int* row_ptr1, const int* col_ind1, const double* values1, int nnz1,
                               const int* row_ind2,const int* row_ptr2, const int* col_ind2, const double* values2, int nnz2,
                               int* row_ptr_out, int* col_ind_out, double* values_out, int num_cols1, int num_cols2,
                               double alpha, double beta){
    // use row_ind to get the row index of each element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < nnz1){
        int start_ind1 = row_ptr1[row_ind1[i]];
        int start_ind2 = row_ptr2[row_ind1[i]];
        col_ind_out[start_ind1 + start_ind2 + (i - start_ind1)] = col_ind1[i];
        values_out[start_ind1 + start_ind2 + (i - start_ind1)] = alpha * values1[i];
    }
    if(i < nnz2){
        int start_ind1 = row_ptr1[row_ind2[i]+1];
        int start_ind2 = row_ptr2[row_ind2[i]];
        col_ind_out[start_ind1 + start_ind2 + (i - start_ind2)] = col_ind2[i] + num_cols1;
        values_out[start_ind1 + start_ind2 + (i - start_ind2)] = beta * values2[i];
    }
    int num_row = row_ind1[nnz1-1] + 1;
    if(i < num_row + 1){
        row_ptr_out[i] = row_ptr1[i] + row_ptr2[i];
    }

}

#endif // SPARSE_STACK_H