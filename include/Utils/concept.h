#ifndef CUOPT_CONCEPT_H
#define CUOPT_CONCEPT_H
#include <cublas_v2.h>   


typedef unsigned int size_s;        // Alias for unsigned int
typedef unsigned long long int size_l;  // Alias for unsigned long long

template <typename T>
struct CudaTypeMapper;

template <>
struct CudaTypeMapper<double> {
    static const cudaDataType value = CUDA_R_64F;
};

template <>
struct CudaTypeMapper<float> {
    static const cudaDataType value = CUDA_R_32F;
};

template <>
struct CudaTypeMapper<int> {
    static const cudaDataType value = CUDA_R_32I;
};

template <>
struct CudaTypeMapper<size_s> {
    static const cudaDataType value = CUDA_R_32U;
};

template <>
struct CudaTypeMapper<size_l> {
    static const cudaDataType value = CUDA_R_64U;
};





#endif // CUOPT_CONCEPT_H
