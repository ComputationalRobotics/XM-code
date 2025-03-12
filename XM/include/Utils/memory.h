#ifndef CUOPT_MEMORY_H
#define CUOPT_MEMORY_H

// this header file contains memory wrappers for:
// dense vec (on device and host), 
// sparse vec (on device), 
// and sparse mat (on device, CSC and CSR format)
// we also provide wrappers for cuda streams, cublas handles, cusolver handles, and cusparse handles

#include <cblas.h>
#include <iostream>
#include <cassert>
#include <algorithm>
#include "check.h"
#include "concept.h"
#include <cstring>
#include <cusolverSp.h> // Add this line to include the cusolverSp header

// cuda stream wrapper
class DeviceStream {
    public:
        size_s gpu_id;
        cudaStream_t stream;

        // Initialize using the index of GPU
        DeviceStream(): gpu_id(0), stream(NULL) {}
        DeviceStream(const size_s gpu_id): gpu_id(gpu_id), stream(NULL) {}

        inline void set_gpu_id(const size_s gpu_id) {
            this->gpu_id = gpu_id;
            return;
        }
        inline void activate() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUDA( cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking) );
        }

        ~DeviceStream() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->stream != NULL) {
                CHECK_CUDA( cudaStreamDestroy(this->stream) );
                this->stream = NULL;
            }
            // std::cout << "DeviceStream destructor called!" << std::endl;
        }
};  

// cublas handle wrapper
class DeviceBlasHandle {
    public:
        size_s gpu_id;
        cublasHandle_t cublas_handle;

        DeviceBlasHandle(): gpu_id(0), cublas_handle(NULL) {}
        DeviceBlasHandle(const size_s gpu_id): gpu_id(gpu_id), cublas_handle(NULL) {}

        inline void set_gpu_id(const size_s gpu_id) {
            this->gpu_id = gpu_id;
            return;
        }
        inline void activate() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasCreate_v2(&this->cublas_handle) );
            return;
        }
        inline void activate(const DeviceStream& device_stream) {
            assert(device_stream.gpu_id == this->gpu_id);
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasCreate_v2(&this->cublas_handle) );
            CHECK_CUBLAS( cublasSetStream_v2(this->cublas_handle, device_stream.stream) );
            return;
        }

        ~DeviceBlasHandle() {
            if (this->cublas_handle != NULL) {
                CHECK_CUBLAS( cublasDestroy_v2(this->cublas_handle) );
                this->cublas_handle = NULL;
            }
            // std::cout << "DeviceBlasHandle destructor called!" << std::endl;
        }
};

// cusolver (dense) handle wrapper
class DeviceSolverDnHandle {
    public:
        size_s gpu_id;
        cusolverDnHandle_t cusolver_dn_handle;

        DeviceSolverDnHandle(): gpu_id(0), cusolver_dn_handle(NULL) {}
        DeviceSolverDnHandle(const size_s gpu_id): gpu_id(gpu_id), cusolver_dn_handle(NULL) {}

        inline void set_gpu_id(const size_s gpu_id) {
            this->gpu_id = gpu_id;
            return;
        }
        inline void activate() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSOLVER( cusolverDnCreate(&this->cusolver_dn_handle) );
            return;
        }
        inline void activate(const DeviceStream& device_stream) {
            assert(device_stream.gpu_id == this->gpu_id);
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSOLVER( cusolverDnCreate(&this->cusolver_dn_handle) );
            CHECK_CUSOLVER( cusolverDnSetStream(this->cusolver_dn_handle, device_stream.stream) );
            return;
        }

        ~DeviceSolverDnHandle() {
            if (this->cusolver_dn_handle != NULL) {
                CHECK_CUSOLVER( cusolverDnDestroy(this->cusolver_dn_handle) );
                this->cusolver_dn_handle = NULL;
            }
            //std::cout << "DeviceSolverDnHandle destructor called!" << std::endl;
        }
};

// cusolver (dense) handle wrapper
class DeviceSolverSpHandle {
    public:
        size_s gpu_id;
        cusolverSpHandle_t cusolver_sp_handle;

        DeviceSolverSpHandle(): gpu_id(0), cusolver_sp_handle(NULL) {}
        DeviceSolverSpHandle(const size_s gpu_id): gpu_id(gpu_id), cusolver_sp_handle(NULL) {}

        inline void set_gpu_id(const size_s gpu_id) {
            this->gpu_id = gpu_id;
            return;
        }
        inline void activate() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSOLVER( cusolverSpCreate(&this->cusolver_sp_handle) );
            return;
        }
        inline void activate(const DeviceStream& device_stream) {
            assert(device_stream.gpu_id == this->gpu_id);
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSOLVER( cusolverSpCreate(&this->cusolver_sp_handle) );
            CHECK_CUSOLVER( cusolverSpSetStream(this->cusolver_sp_handle, device_stream.stream) );
            return;
        }

        ~DeviceSolverSpHandle() {
            if (this->cusolver_sp_handle != NULL) {
                CHECK_CUSOLVER( cusolverSpDestroy(this->cusolver_sp_handle) );
                this->cusolver_sp_handle = NULL;
            }
            //std::cout << "DeviceSolverSpHandle destructor called!" << std::endl;
        }
};

// cusparse handle wrapper
class DeviceSparseHandle {
    public:
        size_s gpu_id;
        cusparseHandle_t cusparse_handle;

        DeviceSparseHandle(): gpu_id(0), cusparse_handle(NULL) {}
        DeviceSparseHandle(const size_s gpu_id): gpu_id(gpu_id), cusparse_handle(NULL) {}

        inline void set_gpu_id(const size_s gpu_id) {
            this->gpu_id = gpu_id;
            return;
        }
        inline void activate() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSPARSE( cusparseCreate(&this->cusparse_handle) );
            return;
        }
        inline void activate(const DeviceStream& device_stream) {
            assert(device_stream.gpu_id == this->gpu_id);
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUSPARSE( cusparseCreate(&this->cusparse_handle) );
            CHECK_CUSPARSE( cusparseSetStream(this->cusparse_handle, device_stream.stream) );
            return;
        }

        ~DeviceSparseHandle() {
            if (this->cusparse_handle != NULL) {
                CHECK_CUSPARSE( cusparseDestroy(this->cusparse_handle) );
                this->cusparse_handle = NULL;
            }
            // std::cout << "DeviceSparseHandle destructor called!" << std::endl;
        }
};

template <typename T>
class HostDnTen {
public:
    size_s* dimensions = nullptr; // Array to store the size of each dimension
    size_s num_dims = 0;    // Number of dimensions
    size_l total_size = 1;
    T* vals = nullptr;         // Pointer to the array of values
    T initial_value;
    bool is_initial_value = false;

    HostDnTen() {}
    
    HostDnTen(const size_s num_dims, const size_s* dimensions) : num_dims(num_dims){
        this->dimensions = (size_s*)malloc(sizeof(size_s) * num_dims);
        std::memcpy(this->dimensions, dimensions, sizeof(size_s) * num_dims); // Copy dimensions
        total_size = 1;
        for (size_s i = 0; i < num_dims; ++i) {
            total_size *= dimensions[i]; // Calculate total number of elements
        }
        this->allocate();
    }
    HostDnTen(const size_s num_dims, const size_s* dimensions, T initial_value) : num_dims(num_dims) ,initial_value(initial_value){
        this->dimensions = (size_s*)malloc(sizeof(size_s) * num_dims);
        std::memcpy(this->dimensions, dimensions, sizeof(size_s) * num_dims); // Copy dimensions
        is_initial_value = true;
        total_size = 1;
        for (size_s i = 0; i < num_dims; ++i) {
            total_size *= dimensions[i]; // Calculate total number of elements
        }
        this->allocate();
    }

    HostDnTen(std::initializer_list<size_s> dims) : num_dims(dims.size()){
        dimensions = new size_s[num_dims];
        std::copy(dims.begin(), dims.end(), dimensions);
        total_size = 1;
        for (size_s i = 0; i < num_dims; ++i) {
            total_size *= dimensions[i]; // Calculate total number of elements
        }
        this->allocate();
    }

    // Allocate memory for the values
    void allocate() {
        if (this->vals == nullptr) {
            this->vals = (T*)malloc(sizeof(T) * total_size);
            if (this->vals && is_initial_value) {
                std::fill(this->vals, this->vals + total_size, initial_value);
            }
        }else{
            std::cout << "Error: Vec already allocated!" << std::endl;
        }
    }

    // Print matrix, if matrix is larger the 3-dimension then donot print
    void print() {
        if(num_dims == 1){
            for (size_s j = 0; j < dimensions[0]; j++) {
                std::printf("%0.2f ", vals[j]);
            }
            std::printf("\n");
        }else if(num_dims == 2){
            for (size_s i = 0; i < dimensions[0]; i++) {
                for (size_s j = 0; j < dimensions[1]; j++) {
                    std::printf("%0.2f ", vals[j * dimensions[0] + i]);
                }
                std::printf("\n");
            }
        }else{
            std::cout << "Matrix dimension larger than 2" << std::endl;
        }
    }

    // Free allocated memory
    ~HostDnTen() {
        if (this->vals != nullptr) {
            free(this->vals);
            this->vals = nullptr;
        }
        if (dimensions != nullptr) {
            free(dimensions); // Don't forget to free dimensions if allocated
            dimensions = nullptr;
        }
        // std::cout << "HostDnTen destructor called!" << std::endl;
    }
};

template <typename T>
__global__ void setValueKernel(T* array, size_t index, T value) {
        array[index] = value;
}

// dense vector wrapper on device
template <typename T>
class DeviceDnTen {
public:
    size_s gpu_id = 0;
    size_s* dimensions = nullptr; // Array to store the size of each dimension
    size_s num_dims = 0;    // Number of dimensions
    size_l total_size = 1;
    T* vals = nullptr;           // Pointer to the array of values
    T* vals_host = nullptr;
    cusparseDnVecDescr_t cusparse_descr = NULL;
    cusparseDnMatDescr_t cusparse_descr_mat = NULL;
    // bool* isdestroied = new bool(false);

    DeviceDnTen(){}

    DeviceDnTen(const size_s num_dims, const size_s* dimensions) 
        : num_dims(num_dims){
        this->dimensions = (size_s*)malloc(sizeof(size_s) * num_dims);
        std::memcpy(this->dimensions, dimensions, sizeof(size_s) * num_dims); // Copy dimensions
        
        // Calculate total size
        total_size = 1;
        for (size_s i = 0; i < num_dims; ++i) {
            total_size *= dimensions[i];
        }

        this->allocate();
    }

    DeviceDnTen(const size_s gpu_id, const size_s num_dims, const size_s* dimensions) 
        : gpu_id(gpu_id), num_dims(num_dims) {
        this->dimensions = (size_s*)malloc(sizeof(size_s) * num_dims);
        std::memcpy(this->dimensions, dimensions, sizeof(size_s) * num_dims); // Copy dimensions
        
        // Calculate total size
        total_size = 1;
        for (size_s i = 0; i < num_dims; ++i) {
            total_size *= dimensions[i];
        }

        this->allocate();
    }

    DeviceDnTen(std::initializer_list<size_s> dims) : num_dims(dims.size()) {
        dimensions = new size_s[num_dims];
        std::copy(dims.begin(), dims.end(), dimensions);

        // Calculate total number of elements
        total_size = 1;
        for (size_s i = 0; i < num_dims; ++i) {
            total_size *= dimensions[i]; 
        }
        this->allocate();
    }

    DeviceDnTen(const DeviceDnTen<T>& other) 
        : gpu_id(other.gpu_id), num_dims(other.num_dims), total_size(other.total_size) {

        this->dimensions = (size_s*)malloc(sizeof(size_s) * num_dims);
        std::memcpy(dimensions, other.dimensions, sizeof(size_s) * num_dims);
        
        this->allocate();
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaMemcpy(vals, other.vals, sizeof(T) * total_size, cudaMemcpyDeviceToDevice);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void copy(const DeviceDnTen<T>& other){
        gpu_id = other.gpu_id;
        num_dims = other.num_dims;
        total_size = other.total_size;

        // TODO: here we need more safe code
        this->dimensions = (size_s*)malloc(sizeof(size_s) * num_dims);
        std::memcpy(dimensions, other.dimensions, sizeof(size_s) * num_dims);
        
        this->allocate();
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaMemcpy(vals, other.vals, sizeof(T) * total_size, cudaMemcpyDeviceToDevice);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    inline void allocate() {

        // std::cout << "allocate with size:  " << this->total_size<<std::endl;
        CHECK_CUDA(cudaSetDevice(this->gpu_id));
        CHECK_CUDA(cudaMalloc((void**)&this->vals, sizeof(T) * total_size));
        CHECK_CUDA(cudaMemset(this->vals, 0, sizeof(T) * total_size));
        if(num_dims == 1){
            CHECK_CUSPARSE(cusparseCreateDnVec(&this->cusparse_descr, total_size, this->vals, CudaTypeMapper<T>::value));
        }
        else if(num_dims == 2){
            CHECK_CUSPARSE(cusparseCreateDnMat(&this->cusparse_descr_mat, dimensions[0],dimensions[1],dimensions[0], this->vals, CudaTypeMapper<T>::value, CUSPARSE_ORDER_COL));
        }
       
        
    }

    void setValueAt(size_s index, T value) {
        setValueKernel<<<1, 1>>>(vals, index, value);
    }

    void SynchronizeDevicetoHost(){
        #ifdef SAFE_MODE
            if(this->vals == nullptr){
                std::cout<<"device hasn't initialized!"<<std::endl;
                return;
            }
        #endif
        if(this->vals_host == nullptr){
            this->vals_host = (T*)malloc(sizeof(T) * total_size);
        }
        // always Synchronize to avoid read-write conflict
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpyAsync(vals_host, vals, sizeof(T) * total_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    void SynchronizeHostToDevice(T* hostTen){
        #ifdef SAFE_MODE
            if(hostTen == nullptr){
                std::cout<<"host hasn't initialized!"<<std::endl;
                return;
            }
        #endif
        if(this->vals == nullptr){
            CHECK_CUDA(cudaSetDevice(this->gpu_id));
            CHECK_CUDA(cudaMalloc((void**)&this->vals, sizeof(T) * total_size));
            // CHECK_CUDA(cudaMemset(this->vals, 0, sizeof(T) * total_size));
            CHECK_CUSPARSE(cusparseCreateDnVec(&this->cusparse_descr, total_size, this->vals, CudaTypeMapper<T>::value));
        }
        // always Synchronize to avoid read-write conflict
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpyAsync( vals, hostTen, sizeof(T) * total_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

     void SynchronizeHostToDevice(){
        #ifdef SAFE_MODE
            if(this->vals_host == nullptr){
                std::cout<<"host hasn't initialized!"<<std::endl;
                return;
            }
        #endif
        if(this->vals == nullptr){
            CHECK_CUDA(cudaSetDevice(this->gpu_id));
            CHECK_CUDA(cudaMalloc((void**)&this->vals, sizeof(T) * total_size));
            // CHECK_CUDA(cudaMemset(this->vals, 0, sizeof(T) * total_size));
            CHECK_CUSPARSE(cusparseCreateDnVec(&this->cusparse_descr, total_size, this->vals, CudaTypeMapper<T>::value));
        }
        // always Synchronize to avoid read-write conflict
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpyAsync( vals, vals_host, sizeof(T) * total_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // norm of GPU vector
    double get_dnorm(const DeviceBlasHandle& cublas_H) {
        double norm = 0;
        CHECK_CUDA(cudaSetDevice(this->gpu_id));
        CHECK_CUBLAS(cublasDnrm2_v2(cublas_H.cublas_handle, this->total_size, this->vals, 1, &norm));
        return norm;
    }

    // Print matrix, if matrix is larger the 3-dimension then donot print
    void print() {
        if(vals_host == nullptr){
            // always Synchronize to avoid read-write conflict
            SynchronizeDevicetoHost();
        }
        if(num_dims == 1){
            for (size_s j = 0; j < dimensions[0]; j++) {
                if(CudaTypeMapper<T>::value == CUDA_R_64F || CudaTypeMapper<T>::value == CUDA_R_32F){
                    std::printf("%1.3e, ", vals_host[j]);
                }
                else{
                    std::printf("%d, ", vals_host[j]);
                }
                
            }
            std::printf("\n");
        }else if(num_dims == 2){
            for (size_s i = 0; i < dimensions[0]; i++) {
                for (size_s j = 0; j < dimensions[1]; j++) {
                    if(CudaTypeMapper<T>::value == CUDA_R_64F || CudaTypeMapper<T>::value == CUDA_R_32F){
                        std::printf("%1.3e, ", vals_host[j * dimensions[0] + i]);
                    }
                    else{
                        std::printf("%d, ", vals_host[j * dimensions[0] + i]);
                    }                   
                }
                std::printf(";\n");
            }
        }else{
            std::cout << "Matrix dimension larger than 2" << std::endl;
        }
        std::printf("\n");
    }

    ~DeviceDnTen() {
        CHECK_CUDA(cudaSetDevice(this->gpu_id));
        if (this->vals != nullptr) {
            CHECK_CUDA(cudaFree(this->vals));
            this->vals = nullptr;
        }
        if (this->cusparse_descr != NULL) {
            CHECK_CUSPARSE(cusparseDestroyDnVec(this->cusparse_descr));
            this->cusparse_descr = NULL;
        }
        if (this->cusparse_descr_mat != NULL) {
            CHECK_CUSPARSE(cusparseDestroyDnMat(this->cusparse_descr_mat));
            this->cusparse_descr_mat = NULL;
        }
        if (dimensions != nullptr) {
            free(dimensions);
            dimensions = nullptr;
        }
        if (vals_host != nullptr) {
            free(vals_host);
            vals_host = nullptr;
        }

    }
};

//sparse matrix wrapper on device: CSC format
template <typename T>
class DeviceSpMatCSC {
    public:
        size_s gpu_id = 0;
        size_s row_size = 0; 
        size_s col_size = 0;
        size_l nnz = 0;
        int* col_ptrs = nullptr;
        int* row_ids = nullptr;
        T* vals = nullptr; 
        cusparseSpMatDescr_t cusparse_descr = nullptr;

        DeviceSpMatCSC(){}
        DeviceSpMatCSC(const size_s gpu_id, const size_s row_size, const size_s col_size, const size_l nnz):
            gpu_id(gpu_id), row_size(row_size), col_size(col_size), nnz(nnz),
            col_ptrs(nullptr), row_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {
                this->allocate();
            }

        inline void allocate() {
            if (this->vals == nullptr) {
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->col_ptrs, sizeof(int) * (this->col_size + 1)) );
                CHECK_CUDA( cudaMalloc((void**) &this->row_ids, sizeof(int) * this->nnz) );
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(T) * this->nnz) );
                CHECK_CUSPARSE( cusparseCreateCsc(
                    &this->cusparse_descr, this->row_size, this->col_size, this->nnz,
                    this->col_ptrs, this->row_ids, this->vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CudaTypeMapper<T>::value
                ) );
            }
            return;
        }

        void SynchronizeHostToDevice(int* host_col_ptrs, int* host_row_ids, T* host_val){
        // always Synchronize to avoid read-write conflict
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpyAsync(col_ptrs, host_col_ptrs, sizeof(int) * (col_size + 1), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyAsync(row_ids, host_row_ids, sizeof(int) * nnz, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyAsync(vals, host_val, sizeof(T) * nnz, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            double norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->nnz, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceSpMatCSC() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->col_ptrs != nullptr) {
                CHECK_CUDA( cudaFree(this->col_ptrs) );
                this->col_ptrs = nullptr;
            }
            if (this->row_ids != nullptr) {
                CHECK_CUDA( cudaFree(this->row_ids) );
                this->row_ids = nullptr;
            }
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroySpMat(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceSpMatDoubleCSC destructor called!" << std::endl;
        }
};

//sparse matrix wrapper on device: CSC format
template <typename T>
class DeviceSpMatCSR {
    public:
        size_s gpu_id = 0;
        size_s row_size = 0; 
        size_s col_size = 0;
        size_l nnz = 0;
        int* col_ids = nullptr;
        int* row_ptrs = nullptr;
        T* vals = nullptr; 
        cusparseSpMatDescr_t cusparse_descr = nullptr;

        DeviceSpMatCSR(){}
        DeviceSpMatCSR(const size_s gpu_id, const size_s row_size, const size_s col_size, const size_l nnz):
            gpu_id(gpu_id), row_size(row_size), col_size(col_size), nnz(nnz),
            col_ids(nullptr), row_ptrs(nullptr), vals(nullptr), cusparse_descr(NULL) {
                this->allocate();
            }

        inline void allocate() {
            if (this->vals == nullptr) {
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->row_ptrs, sizeof(int) * (this->row_size + 1)) );
                if (nnz > 0) {
                    CHECK_CUDA( cudaMalloc((void**) &this->col_ids, sizeof(int) * this->nnz) );
                    CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(T) * this->nnz) );
                } else {
                    this->col_ids = nullptr;
                    this->vals = nullptr;
                }
                CHECK_CUSPARSE( cusparseCreateCsr(
                    &this->cusparse_descr, this->row_size, this->col_size, this->nnz,
                    this->row_ptrs, this->col_ids, this->vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CudaTypeMapper<T>::value
                ) );
            }
        }

        void SynchronizeHostToDevice(int* host_col_ids, int* host_row_ptrs, T* host_val){
            // always Synchronize to avoid read-write conflict
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpyAsync(col_ids, host_col_ids, sizeof(int) * this->nnz, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpyAsync(row_ptrs, host_row_ptrs, sizeof(int) * (this->row_size+1), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpyAsync(vals, host_val, sizeof(T) * this->nnz, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        void SynchronizeDeviceToDevice(int* device_col_ids, int* device_row_ptrs, T* device_val){
            // always Synchronize to avoid read-write conflict
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpyAsync(col_ids, device_col_ids, sizeof(int) * this->nnz, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(row_ptrs, device_row_ptrs, sizeof(int) * (this->row_size+1), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(vals, device_val, sizeof(T) * this->nnz, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            double norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->nnz, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceSpMatCSR() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->col_ids != nullptr) {
                CHECK_CUDA( cudaFree(this->col_ids) );
                this->col_ids = nullptr;
            }
            if (this->row_ptrs != nullptr) {
                CHECK_CUDA( cudaFree(this->row_ptrs) );
                this->row_ptrs = nullptr;
            }
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroySpMat(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceSpMatDoubleCSC destructor called!" << std::endl;
        }
};

//sparse matrix wrapper on device: COO format
template <typename T>
class DeviceSpMatCOO {
    public:
        size_s gpu_id = 0;
        size_s row_size = 0; 
        size_s col_size = 0;
        size_l nnz = 0;
        int* col_ids = nullptr;
        int* row_ids = nullptr;
        T* vals = nullptr; 
        cusparseSpMatDescr_t cusparse_descr = nullptr;

        DeviceSpMatCOO(){}
        DeviceSpMatCOO(const size_s gpu_id, const size_s row_size, const size_s col_size, const size_l nnz):
            gpu_id(gpu_id), row_size(row_size), col_size(col_size), nnz(nnz),
            col_ids(nullptr), row_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {
                this->allocate();
            }

        inline void allocate() {
            if (this->vals == nullptr) {
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                
                if (nnz > 0) {
                    CHECK_CUDA( cudaMalloc((void**) &this->row_ids, sizeof(int) * this->nnz) );
                    CHECK_CUDA( cudaMalloc((void**) &this->col_ids, sizeof(int) * this->nnz) );
                    CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(T) * this->nnz) );
                } else {
                    this->row_ids = nullptr;
                    this->col_ids = nullptr;
                    this->vals = nullptr;
                }
                CHECK_CUSPARSE( cusparseCreateCoo(
                    &this->cusparse_descr, this->row_size, this->col_size, this->nnz,
                    this->row_ids, this->col_ids, this->vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CudaTypeMapper<T>::value
                ) );
            }
        }

        void SynchronizeHostToDevice(int* host_col_ids, int* host_row_ids, T* host_val){
            // always Synchronize to avoid read-write conflict
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpyAsync(col_ids, host_col_ids, sizeof(int) * this->nnz, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpyAsync(row_ids, host_row_ids, sizeof(int) * this->nnz, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpyAsync(vals, host_val, sizeof(T) * nnz, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            double norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->nnz, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceSpMatCOO() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->col_ids != nullptr) {
                CHECK_CUDA( cudaFree(this->col_ids) );
                this->col_ids = nullptr;
            }
            if (this->row_ids != nullptr) {
                CHECK_CUDA( cudaFree(this->row_ids) );
                this->row_ids = nullptr;
            }
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroySpMat(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceSpMatDoubleCSC destructor called!" << std::endl;
        }
};


//////////////////////////////////////////////
//Shucheng's code



// dense vector wrapper on host: double type
class HostDnVecDouble {
    public:
        int size;
        double* vals;

        HostDnVecDouble(): size(0), vals(nullptr) {}
        HostDnVecDouble(const int size, bool as_byte = false): size(size), vals(nullptr) {
            this->allocate(size, as_byte);
        }

        inline void allocate(const int size, bool as_byte = false) {
            if (this->vals == nullptr) {
                if (as_byte) {
                    this->size = (size + sizeof(double) - 1) / sizeof(double);
                } else {
                    this->size = size;
                }
                this->vals = (double*) malloc(sizeof(double) * this->size);
            }
            return;
        }
        inline double get_norm() {
            return cblas_dnrm2(this->size, this->vals, 1);
        }

        ~HostDnVecDouble() {
            if (this->vals != nullptr) {
                free(this->vals);
                this->vals = nullptr;
            }
            // std::cout << "HostDnVecDouble destructor called!" << std::endl;
        }
};

// dense vector wrapper on host: float type
class HostDnVecFloat {
    public:
        int size;
        float* vals;

        HostDnVecFloat(): size(0), vals(nullptr) {}
        HostDnVecFloat(const int size, bool as_byte = false): size(size), vals(nullptr) {
            this->allocate(size, as_byte);
        }

        inline void allocate(const int size, bool as_byte = false) {
            if (this->vals == nullptr) {
                if (as_byte) {
                    this->size = (size + sizeof(float) - 1) / sizeof(float);
                } else {
                    this->size = size;
                }
                this->vals = (float*) malloc(sizeof(float) * this->size);
            }
            return;
        }
        inline float get_norm() {
            return cblas_snrm2(this->size, this->vals, 1);
        }

        ~HostDnVecFloat() {
            if (this->vals != nullptr) {
                free(this->vals);
                this->vals = nullptr;
            }
            // std::cout << "HostDnVecFloat destructor called!" << std::endl;
        }
};

// dense vector wrapper on host: int type
class HostDnVecInt {
    public:
        int size;
        int* vals;

        HostDnVecInt(): size(0), vals(nullptr) {}
        HostDnVecInt(const int size): size(size), vals(nullptr) {
            this->allocate(this->size);
        }

        inline void allocate(const int size) {
            if (this->vals == nullptr) {
                this->size = size;
                this->vals = (int*) malloc(sizeof(int) * size);
            }
            return;
        }

        ~HostDnVecInt() {
            if (this->vals != nullptr) {
                free(this->vals);
                this->vals = nullptr;
            }
            // std::cout << "HostDnVecInt destructor called!" << std::endl;
        }
};

// dense vector wrapper on host: size_t type
class HostDnVecLongInt {
    public:
        int size;
        size_t* vals;

        HostDnVecLongInt(): size(0), vals(nullptr) {}
        HostDnVecLongInt(const int size): size(size), vals(nullptr) {
            this->allocate(this->size);
        }

        inline void allocate(const int size) {
            if (this->vals == nullptr) {
                this->size = size;
                this->vals = (size_t*) malloc(sizeof(size_t) * size);
            }
            return;
        }

        ~HostDnVecLongInt() {
            if (this->vals != nullptr) {
                free(this->vals);
                this->vals = nullptr;
            }
            // std::cout << "HostDnVecLongInt destructor called!" << std::endl;
        }
};

// dense vector wrapper on host: ptrdiff_t type
class HostDnVecPtrdiff_t {
    public:
        size_t size;
        ptrdiff_t* vals;

        HostDnVecPtrdiff_t(): size(0), vals(nullptr) {}
        HostDnVecPtrdiff_t(const size_t size): size(size), vals(nullptr) {
            this->allocate(this->size);
        }

        inline void allocate(const size_t size) {
            if (this->vals == nullptr) {
                this->size = size;
                this->vals = (ptrdiff_t*) malloc(sizeof(ptrdiff_t) * size);
            }
            return;
        }

        ~HostDnVecPtrdiff_t() {
            if (this->vals != nullptr) {
                free(this->vals);
                this->vals = nullptr;
            }
            // std::cout << "HostDnVecPtrdiff_t destructor called!" << std::endl;
        }
};

// dense vector wrapper on device: double type
class DeviceDnVecDouble {
    public:
        int gpu_id;
        int size;
        double* vals;
        cusparseDnVecDescr_t cusparse_descr;

        DeviceDnVecDouble(): gpu_id(0), size(0), vals(nullptr), cusparse_descr(NULL) {}
        DeviceDnVecDouble(const int gpu_id, const int size, bool as_byte = false):
            gpu_id(gpu_id), size(size), vals(nullptr), cusparse_descr(NULL) {
            this->allocate(this->gpu_id, this->size, as_byte);
        }

        inline void allocate(const int gpu_id, const int size, bool as_byte = false) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->size = size;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                // as_byte is used to allocate buffer size, which is usually given in terms of bytes
                if (as_byte) {
                    this->size = (size + sizeof(double) - 1) / sizeof(double);
                } else {
                    this->size = size;
                }
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(double) * this->size) );
                CHECK_CUSPARSE( cusparseCreateDnVec(&this->cusparse_descr, this->size, this->vals, CUDA_R_64F) );
            }
            return;
        }
        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            double norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->size, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceDnVecDouble() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroyDnVec(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceDnVecDouble destructor called!" << std::endl;
        }
};

// dense vector wrapper on device: float type
class DeviceDnVecFloat {
    public:
        int gpu_id;
        int size;
        float* vals;
        cusparseDnVecDescr_t cusparse_descr;

        DeviceDnVecFloat(): gpu_id(0), size(0), vals(nullptr), cusparse_descr(NULL) {}
        DeviceDnVecFloat(const int gpu_id, const int size, bool as_byte = false):
            gpu_id(gpu_id), size(size), vals(nullptr), cusparse_descr(NULL) {
            this->allocate(this->gpu_id, this->size, as_byte);
        }

        inline void allocate(const int gpu_id, const int size, bool as_byte = false) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                // as_byte is used to allocate buffer size, which is usually given in terms of bytes
                if (as_byte) {
                    this->size = (size + sizeof(float) - 1) / sizeof(float);
                } else {
                    this->size = size;
                }
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(float) * this->size) );
                CHECK_CUSPARSE( cusparseCreateDnVec(&this->cusparse_descr, this->size, this->vals, CUDA_R_32F) );
            }
            return;
        }
        inline float get_norm(const DeviceBlasHandle& cublas_H) {
            float norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasSnrm2_v2(
                cublas_H.cublas_handle, this->size, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceDnVecFloat() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroyDnVec(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceDnVecFloat destructor called!" << std::endl;
        }
};

// dense vector wrapper on device: int type
class DeviceDnVecInt {
    public:
        int gpu_id;
        int size;
        int* vals;

        DeviceDnVecInt(): gpu_id(0), size(0), vals(nullptr) {}
        DeviceDnVecInt(const int gpu_id, const int size): gpu_id(gpu_id), size(size), vals(nullptr) {
            this->allocate(this->gpu_id, this->size);
        }

        inline void allocate(const int gpu_id, const int size) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->size = size;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(int) * this->size) );
            }
            return;
        }

        ~DeviceDnVecInt() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            // std::cout << "DeviceDnVecInt destructor called!" << std::endl;
        }
};

// dense vector wrapper on device: long int type
class DeviceDnVecLongInt {
    public:
        int gpu_id;
        int size;
        size_t* vals;

        DeviceDnVecLongInt(): gpu_id(0), size(0), vals(nullptr) {}
        DeviceDnVecLongInt(const int gpu_id, const int size): gpu_id(gpu_id), size(size), vals(nullptr) {
            this->allocate(this->gpu_id, this->size);
        }

        inline void allocate(const int gpu_id, const int size) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->size = size;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(size_t) * this->size) );
            }
            return;
        }

        ~DeviceDnVecLongInt() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            // std::cout << "DeviceDnVecLongInt destructor called!" << std::endl;
        }
};

// sparse vector wrapper on device: double type
class DeviceSpVecDouble {
    public:
        int gpu_id;
        int size; 
        int nnz;
        int* indices;
        double* vals; 
        cusparseSpVecDescr_t cusparse_descr;

        DeviceSpVecDouble(): gpu_id(0), size(0), nnz(0), 
            indices(nullptr), vals(nullptr), cusparse_descr(NULL) {}
        DeviceSpVecDouble(const int gpu_id, const int size, const int nnz):
            gpu_id(gpu_id), size(size), nnz(nnz), 
            vals(nullptr), indices(nullptr), cusparse_descr(NULL) {
                this->allocate(gpu_id, size, nnz);
            }
        
        inline void allocate(const int gpu_id, const int size, const int nnz) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->size = size; 
                this->nnz = nnz;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->indices, sizeof(int) * this->nnz) );
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(double) * this->nnz) );
                CHECK_CUSPARSE( cusparseCreateSpVec(
                    &this->cusparse_descr, this->size, this->nnz, 
                    this->indices, this->vals, 
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F 
                ) );
            }
            return;
        }

        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            double norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->nnz, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceSpVecDouble() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->indices != nullptr) {
                CHECK_CUDA( cudaFree(this->indices) );
                this->indices = nullptr;
            }
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroySpVec(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceSpVecDouble destructor called!" << std::endl;
        }
};

// sparse vector wrapper on device: double type + CSC format
class DeviceSpMatDoubleCSC {
    public:
        int gpu_id;
        int row_size; 
        int col_size;
        int nnz;
        int* col_ptrs;
        int* row_ids;
        double* vals; 
        cusparseSpMatDescr_t cusparse_descr;

        DeviceSpMatDoubleCSC(): gpu_id(0), row_size(0), col_size(0), nnz(0),
            col_ptrs(nullptr), row_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {}
        DeviceSpMatDoubleCSC(const int gpu_id, const int row_size, const int col_size, const int nnz):
            gpu_id(gpu_id), row_size(row_size), col_size(col_size), nnz(nnz),
            col_ptrs(nullptr), row_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {
                this->allocate(gpu_id, row_size, col_size, nnz);
            }

        inline void allocate(const int gpu_id, const int row_size, const int col_size, const int nnz) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->row_size = row_size;
                this->col_size = col_size;
                this->nnz = nnz;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->col_ptrs, sizeof(int) * (this->col_size + 1)) );
                CHECK_CUDA( cudaMalloc((void**) &this->row_ids, sizeof(int) * this->nnz) );
                CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(double) * this->nnz) );
                CHECK_CUSPARSE(cusparseCreateCsc(
                    &this->cusparse_descr, this->row_size, this->col_size, this->nnz,
                    this->col_ptrs, this->row_ids, this->vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
                ) );
            }
            return;
        }

        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            double norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->nnz, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceSpMatDoubleCSC() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->col_ptrs != nullptr) {
                CHECK_CUDA( cudaFree(this->col_ptrs) );
                this->col_ptrs = nullptr;
            }
            if (this->row_ids != nullptr) {
                CHECK_CUDA( cudaFree(this->row_ids) );
                this->row_ids = nullptr;
            }
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroySpMat(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceSpMatDoubleCSC destructor called!" << std::endl;
        }
};

// sparse vector wrapper on device: double type + CSR format
class DeviceSpMatDoubleCSR {
    public:
        int gpu_id;
        int64_t row_size; 
        int64_t col_size;
        int64_t nnz;
        int* row_ptrs;
        int* col_ids;
        double* vals; 
        cusparseSpMatDescr_t cusparse_descr;

        DeviceSpMatDoubleCSR(): gpu_id(0), row_size(0), col_size(0), nnz(0),
            row_ptrs(nullptr), col_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {}
        DeviceSpMatDoubleCSR(const int gpu_id, const int row_size, const int col_size, const int nnz):
            gpu_id(gpu_id), row_size(row_size), col_size(col_size), nnz(nnz),
            row_ptrs(nullptr), col_ids(nullptr), vals(nullptr), cusparse_descr(NULL) {
                this->allocate(gpu_id, row_size, col_size, nnz);
            }

        inline void allocate(const int gpu_id, const int row_size, const int col_size, const int nnz) {
            if (this->vals == nullptr) {
                this->gpu_id = gpu_id;
                this->row_size = row_size;
                this->col_size = col_size;
                this->nnz = nnz;
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaSetDevice(this->gpu_id) );
                CHECK_CUDA( cudaMalloc((void**) &this->row_ptrs, sizeof(int) * (this->row_size + 1)) );
                if (nnz > 0) {
                    CHECK_CUDA( cudaMalloc((void**) &this->col_ids, sizeof(int) * this->nnz) );
                    CHECK_CUDA( cudaMalloc((void**) &this->vals, sizeof(double) * this->nnz) );
                } else {
                    this->col_ids = nullptr;
                    this->vals = nullptr;
                }
                CHECK_CUSPARSE( cusparseCreateCsr(
                    &this->cusparse_descr, this->row_size, this->col_size, this->nnz,
                    this->row_ptrs, this->col_ids, this->vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
                ) );
            }
            return;
        }

        inline double get_norm(const DeviceBlasHandle& cublas_H) {
            double norm;
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            CHECK_CUBLAS( cublasDnrm2_v2(
                cublas_H.cublas_handle, this->nnz, this->vals, 1, &norm
            ) );
            return norm;
        }

        ~DeviceSpMatDoubleCSR() {
            CHECK_CUDA( cudaSetDevice(this->gpu_id) );
            if (this->row_ptrs != nullptr) {
                CHECK_CUDA( cudaFree(this->row_ptrs) );
                this->row_ptrs = nullptr;
            }
            if (this->col_ids != nullptr) {
                CHECK_CUDA( cudaFree(this->col_ids) );
                this->col_ids = nullptr;
            }
            if (this->vals != nullptr) {
                CHECK_CUDA( cudaFree(this->vals) );
                this->vals = nullptr;
            }
            if (this->cusparse_descr != NULL) {
                CHECK_CUSPARSE( cusparseDestroySpMat(this->cusparse_descr) );
                this->cusparse_descr = NULL;
            }
            // std::cout << "DeviceSpMatDoubleCSR destructor called!" << std::endl;
        }
};

#endif