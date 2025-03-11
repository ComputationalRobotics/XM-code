#ifndef BATCHEDQR_H
#define BATCHEDQR_H

#include <vector>
#include <Utils/memory.h>
#include <iostream>
//template
template <typename T>
__global__ void batchedQRKernel(T* d_Q, T* d_A, size_s rows, size_s cols, size_s batchSize) {
    size_s batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx < batchSize){
        // performe a 3-col QR decomposition by hand
        // get the pointer to the current batch
        T* A_c = d_A + batchIdx * rows * cols;
        T* Q_c = d_Q + batchIdx * rows * cols;
        // get the pointer to the first column of the current batch
        // for(int i = 0; i < cols; ++i){
        //     for(int j = 0; j < rows; ++j){
        //         Q_c[i*rows + j] = A_c[i*rows + j];
        //     }
        //     for(int j = 0; j < i; ++j){
        //         T uudot = 0;
        //         T uadot = 0;    
        //         for(int k = 0; k < rows; ++k){
        //             uadot += Q_c[j*rows + k] * A_c[i*rows + k];
        //         }
        //         for(int k = 0; k < rows; ++k){
        //             Q_c[i*rows + k] -= uadot * Q_c[j*rows + k];
                    
        //         }
                
        //     }
        //     T qqdot = 0;
        //     for(int k = 0; k < rows; ++k){
        //         qqdot += Q_c[i*rows + k] * Q_c[i*rows + k];
        //     }
        //     qqdot = sqrt(qqdot);
        //     for(int k = 0; k < rows; ++k){
        //         Q_c[i*rows + k] /= qqdot;
        //     }
        // }
        for(int i = 0; i < cols; ++i){
            for(int j = 0; j < rows; ++j){
                Q_c[i*rows + j] = A_c[i*rows + j];
            }
        }
        for(int i = 0; i < cols; ++i){
            T qqdot = 0;
            for(int k = 0; k < rows; ++k){
                qqdot += Q_c[i*rows + k] * Q_c[i*rows + k];
            }
            qqdot = sqrt(qqdot);
            //printf("qqdot: %1.20e %d\n", qqdot, i);
            for(int k = 0; k < rows; ++k){
                Q_c[i*rows + k] /= qqdot;
            }
            for(int j = i+1; j < cols; ++j){
                T uudot = 0;
                for(int k = 0; k < rows; ++k){
                    uudot += Q_c[i*rows + k] * Q_c[j*rows + k];
                }
                //printf("uudot: %1.20e %d %d\n", uudot, i, j);
                for(int k = 0; k < rows; ++k){
                    Q_c[j*rows + k] -= uudot * Q_c[i*rows + k];
                }
            }
        }
    }
}

// cpu function of batchedQR
template <typename T>
void batchedQR(
    DeviceDnTen<T>& d_Q, DeviceDnTen<T>& d_A, size_s cols, size_s batchSize,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
) {
    int num_block = (batchSize + block_size - 1) / block_size;
    batchedQRKernel<<<num_block, block_size, 0, stream>>>(
        d_Q.vals, d_A.vals, d_A.dimensions[0], cols, batchSize
    );
    return;
}

#endif // BATCHEDQR_H