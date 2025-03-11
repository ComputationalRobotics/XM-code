#ifdef __INTEL_COMPILER
#pragma diag_suppress 20011
#endif

#include <iostream>
#include <Utils/memory.h>
#include <Dense/matmul.h>
#include <Dense/trace.h>
#include <Dense/transpose.h>
#include <Dense/matdiagmul.h>
#include <Dense/matdot.h>
#include <Dense/matdivide.h>
#include <Dense/batchedQR.h>
#include <Dense/eig.h>
#include <Sparse/spmatmul.h>
#include <Sparse/sparseformat.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <chrono>
#include <vector>
#include <fstream>
#include <Optimization/optimization.h>
#include <cusolverSp.h>

using namespace std::chrono;

template <typename T>
__global__ void ConstructZmatrixKernal(T* Z, T* sR, size_s n,size_s o,double lam){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        double norm = 0;
        for(int j = 0; j<o; ++j){
            norm += sR[j*3*n+3*i] * sR[j*3*n+3*i];
        }
        Z[3*i*3*n+3*i] += 2 * lam * (norm - 1);
    }  
}

bool checkeig(opt_var &C, const opt_var &sR, const double lam, opt_var &v, double primal_value){
    // initialize
    DeviceBlasHandle CUOPT_blas_handle;
    CUOPT_blas_handle.activate();
    DeviceSolverSpHandle CUOPT_cusloversp_handle;
    CUOPT_cusloversp_handle.activate();
    DeviceSolverDnHandle CUOPT_cusloverdn_handle;
    CUOPT_cusloverdn_handle.activate();
    DeviceSparseHandle CUOPT_sparse_handle;
    CUOPT_sparse_handle.activate();

    size_s n = C.dimensions[0]/3;
    size_s o = sR.dimensions[1];

    std::vector<int> Acellrow((5*n+1)*2*o,0);
    std::vector<int> Acellcol((5*n+1)*2*o,0);
    std::vector<int> AcellcolCSC(5*n+2,0);
    std::vector<datatype> Acellval((5*n+1)*2*o,0);

    // CONSTRUCTING THE LEFT MATRIX 
    // Left = [];
    // % new try
    // for i = 1:order
    //     V = Acell'*sR_est(i,:)';
    //     V = reshape(V,3*n,[]);
    //     Left = [Left;V];
    // end
    size_l count = 0;
    size_l nnz = 0;
    for(size_s i = 0; i<3; ++i){
        for(size_s j = i; j<3; ++j){
            if(i==j){
                // [i,i,1]
                for(size_s k = 0; k<o; ++k){
                    Acellrow[nnz] = i + k*3*n;
                    Acellcol[nnz] = count;
                    Acellval[nnz] = sR.vals_host[i + k*3*n];
                    nnz += 1;
                }
            }else{
                // [i,j,0.5] and [j,i,0.5]
                for(size_s k = 0; k<o; ++k){
                    Acellrow[nnz] = i + k*3*n;
                    Acellcol[nnz] = count;
                    Acellval[nnz] = sR.vals_host[j + k*3*n] * 0.5;
                    nnz += 1;
                    Acellrow[nnz] = j + k*3*n;
                    Acellcol[nnz] = count;
                    Acellval[nnz] = sR.vals_host[i + k*3*n] * 0.5;
                    nnz += 1;
                    // j >= i
                }
            }
            count += 1;
            AcellcolCSC[count] = nnz;
        }
    }

    for(size_s i = 1; i < n; ++i){
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+1 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+1 + k*3*n] * -0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i+1 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+1 + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+2 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+2 + k*3*n] * -0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+1 + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+1 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i + k*3*n] * 0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+2 + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+2 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i + k*3*n] * 0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i+1 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+2 + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+2 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+1 + k*3*n] * 0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
    }
    

    // for(int i = count + 1; i<5*n+1; ++i){
    //     AcellcolCSC[i] = nnz;
    // }

    DeviceSpMatCSC<double> AcellCSC(0,3*o*n,5*n+1,nnz);
    DeviceSpMatCSR<double> AcellCSR(0,3*o*n,5*n+1,nnz);
    AcellCSC.SynchronizeHostToDevice(AcellcolCSC.data(),Acellrow.data(),Acellval.data());
    CSC2CSR(CUOPT_sparse_handle,AcellCSC,AcellCSR);

    // Z = C;
    //         for i = 1:n-1
    //             Z(3*i+1,3*i+1) = Z(3*i+1,3*i+1) + 2 * lam * (X(3*i+1,3*i+1)-1);
    //         end
    // Right = Z*sR_est';
    // Right = Right(:);
    opt_var Z(C);
    ConstructZmatrixKernal<<<(n + 1024 - 1) / 1024,1024>>>(Z.vals,sR.vals,n,o,lam);
    opt_var Right({3*n,o});
    DnMatDnMat(CUOPT_blas_handle,Right,Z,sR);
    Z.SynchronizeDevicetoHost();
    Right.SynchronizeDevicetoHost();

    // test: sparse multiplication ATA on CUDA
    
    // test: directly use eigen

    Eigen::SparseMatrix<double> Acell_Eigen(3 * o * n, 5 * n + 1);
    Eigen::VectorXd b(3 * o * n), y;

    memcpy(b.data(), Right.vals_host, 3 * o * n * sizeof(double));

    Acell_Eigen.resizeNonZeros(nnz);
    memcpy(Acell_Eigen.valuePtr(), Acellval.data(), nnz * sizeof(double));
    memcpy(Acell_Eigen.innerIndexPtr(), Acellrow.data(), nnz * sizeof(int));
    memcpy(Acell_Eigen.outerIndexPtr(), AcellcolCSC.data(), (5 * n + 2) * sizeof(int));

    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;

    auto start_decomp = std::chrono::high_resolution_clock::now();
    solver.compute(Acell_Eigen);
    auto end_decomp = std::chrono::high_resolution_clock::now();

    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return -1;
    }

    auto start_solve = std::chrono::high_resolution_clock::now();
    y = solver.solve(b);
    auto end_solve = std::chrono::high_resolution_clock::now();

    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed" << std::endl;
        return -1;
    }

    auto decomp_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_decomp - start_decomp);
    auto solve_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_solve - start_solve);

    std::cout << "Decomposition time: " << decomp_duration.count() << " microseconds" << std::endl;
    std::cout << "Solve time: " << solve_duration.count() << " microseconds" << std::endl;

    
    // opt_var y({5*n+1});
    // DeviceDnTen<int> p({5*n+1});
    // cusparseMatDescr_t descrAcell;
    // cusparseCreateMatDescr(&descrAcell);
    // CHECK_CUSPARSE( cusparseSetMatIndexBase(descrAcell, CUSPARSE_INDEX_BASE_ZERO) );
    // CHECK_CUSPARSE( cusparseSetMatType(descrAcell, CUSPARSE_MATRIX_TYPE_GENERAL) );
    // CHECK_CUSPARSE( cusparseSetMatDiagType(descrAcell,CUSPARSE_DIAG_TYPE_NON_UNIT) );
    // // TURN all vraiable to host
    // double *h_vals = (double*) malloc(nnz * sizeof(double));
    // int *h_row_ptrs = (int*) malloc((3*o*n + 1) * sizeof(int));
    // int *h_col_ids = (int*) malloc(nnz * sizeof(int));
    // double *h_right_vals = (double*) malloc(3*o*n * sizeof(double));
    // // Copy AcellCSR.vals (now avell)
    // cudaMemcpy(h_vals, AcellCSR.vals, nnz * sizeof(double), cudaMemcpyDeviceToHost);

    // // Copy AcellCSR.row_ptrs
    // cudaMemcpy(h_row_ptrs, AcellCSR.row_ptrs, (3*o*n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // // Copy AcellCSR.col_ids
    // cudaMemcpy(h_col_ids, AcellCSR.col_ids, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    // // Copy Right.vals
    // cudaMemcpy(h_right_vals, Right.vals, 3*o*n * sizeof(double), cudaMemcpyDeviceToHost);

    // std::vector<double> y(5*n+1,0);
    // std::vector<int> p(5*n+1,0);

    // auto start = std::chrono::high_resolution_clock::now();
    // CHECK_CUSOLVER( cusolverSpDcsrlsqvqrHost(CUOPT_cusloversp_handle.cusolver_sp_handle,3*o*n,5*n+1,nnz,descrAcell,h_vals,h_row_ptrs,h_col_ids,h_right_vals,1e-15,&rankLeft,y.data(),p.data(),&minnorm));
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "cusolverSpDcsrlsqvqrHost execution time: " << duration.count() << " microseconds" << std::endl;

    // std::cout<<"Minnorm of Least-square is: "<<minnorm<<std::endl;
    count = 0;
    nnz = 0;
    for(size_s i = 0; i<3; ++i){
        for(size_s j = i; j<3; ++j){
            if(i==j){
                // [i,i,1]
                Z.vals_host[i + j*3*n] -= y[count];
            }else{
                // [i,j,0.5] and [j,i,0.5]
                Z.vals_host[i + j*3*n] -= y[count] * 0.5;
                Z.vals_host[j + i*3*n] -= y[count] * 0.5;
            }
            count += 1;
        }
    }

    for(size_s i = 1; i < n; ++i){
        Z.vals_host[3*i*3*n + 3*i] -= y[count] * 0.5;
        Z.vals_host[(3*i+1)*3*n + (3*i+1)] -= y[count] * -0.5;
        count += 1;
        
        Z.vals_host[(3*i+1)*3*n + (3*i+1)] -= y[count] * 0.5;
        Z.vals_host[(3*i+2)*3*n + (3*i+2)] -= y[count] * -0.5;
        count += 1;
        
        Z.vals_host[3*i*3*n + 3*i+1] -= y[count] * 0.5;
        Z.vals_host[(3*i+1)*3*n + 3*i] -= y[count] * 0.5;
        count += 1;

        Z.vals_host[3*i*3*n + 3*i+2] -= y[count] * 0.5;
        Z.vals_host[(3*i+2)*3*n + 3*i] -= y[count] * 0.5;
        count += 1;

        Z.vals_host[(3*i+1)*3*n + 3*i+2] -= y[count] * 0.5;
        Z.vals_host[(3*i+2)*3*n + 3*i+1] -= y[count] * 0.5;
        count += 1;
    }
    Z.SynchronizeHostToDevice();

    // eigen value of z
    SingleEigParam eigparam;
    size_t buffer_size_eig = 0;
    size_t buffer_size_host_eig = 0;
    opt_var W({3*n});
    single_eig_get_buffersize_cusolver(CUOPT_cusloverdn_handle,eigparam,Z,W,3*n,&buffer_size_eig,&buffer_size_host_eig);
    //printf("buffer_size_eig: %ld\n",buffer_size_eig);
    void* buffer_eig = nullptr;
    void* buffer_host_eig = (void*)malloc(buffer_size_host_eig);
    int* info = nullptr;
    CHECK_CUDA(cudaMalloc(&buffer_eig,buffer_size_eig));
    CHECK_CUDA(cudaMalloc(&info,sizeof(int)));
    
    single_eig_cusolver(CUOPT_cusloverdn_handle,eigparam,Z,W,buffer_eig,buffer_host_eig,info,3*n,buffer_size_eig,buffer_size_host_eig);
    W.SynchronizeDevicetoHost();
    printf("The min eig is: %1.3e \n",W.vals_host[0]);
    CHECK_CUDA(cudaMemcpy(v.vals,Z.vals,v.total_size * sizeof(datatype),cudaMemcpyDeviceToDevice));
    
    // calculating optimility gap
    std::cout<<"Primal value: "<<primal_value<<std::endl;
    double dual_value = y[0] + y[3] + y[5];
    std::cout<<"Dual value: "<<dual_value<<std::endl;
    std::vector<double> xii(n,0);
    for(size_s i = 0; i<n; ++i){
        for(size_s j = 0; j<o; ++j){
            xii[i] += sR.vals_host[3*i+3*j*n] * sR.vals_host[3*i+3*j*n];
        }
    }
    for(size_s i = 0; i<n; ++i){
        dual_value += (1 - xii[i]*xii[i])*lam;
    }
    std::cout<<"new dual"<<dual_value<<std::endl;
    double bar_s = 1;
    double K = 3*n*bar_s*bar_s;
    double gap = primal_value - dual_value - K*min(0.0,W.vals_host[0]);
    std::cout<<"Optimility gap: "<<gap<<std::endl;

    CHECK_CUDA(cudaFree(buffer_eig));
    free(buffer_host_eig);
    CHECK_CUDA(cudaFree(info));
    // free(h_vals);
    // free(h_row_ptrs);
    // free(h_col_ids);
    // free(h_right_vals);

    // cusparseDestroyMatDescr(descrAcell);

    double min_eig_bound = 1e-4;
    if(n > 2000){
        min_eig_bound = 1e-3;
    }
    else if(n > 5000){
        min_eig_bound = 1e-1;
    }
    else if(n > 10000){
        min_eig_bound = 100;
    }

    if(gap/primal_value < 1e-3 || W.vals_host[0] > -min_eig_bound){
        std::cout<<"BM finished with rank "<< o <<std::endl;
        return true;
    }
    else{
        std::cout<<"BM order plus one"<<std::endl;
        CHECK_CUDA(cudaMemcpy(v.vals,Z.vals,v.total_size * sizeof(datatype),cudaMemcpyDeviceToDevice));
        return false;
    }
    
    // opt_var X({5*n+1});
    // std::vector<datatype> X_h(5*n+1,0);
    // //create rand dense vector
    // for(size_s i = 0; i<5*n+1; ++i){
    //     X_h[i] = i + 1.0;
    // }
    // X.SynchronizeHostToDevice(X_h.data());
    // X.print();
    // opt_var Y({3*o*n});

    // size_t buffer_size = x`_get_buffersize_cusparse(CUOPT_sparse_handle,AcellCSR,X,Y,1.0,0.0);
    // void* buffer = nullptr;  
    // cudaMalloc(&buffer,buffer_size);
    // SpMatDnVec(CUOPT_sparse_handle,AcellCSR,X,Y,1.0,0.0,buffer);
    // CHECK_CUDA(cudaDeviceSynchronize());
    // Y.print();

    
}