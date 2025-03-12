#include <iostream>
#include <Utils/memory.h>
#include <Dense/matmul.h>
#include <Dense/trace.h>
#include <Dense/transpose.h>
#include <Dense/matdiagmul.h>
#include <Dense/matdot.h>
#include <Dense/matdivide.h>
#include <Dense/batchedQR.h>

#include <chrono>
#include <vector>
#include <fstream>
#include <Optimization/optimization.h>

using namespace std::chrono;

template <typename T>
__global__ void positiveManifoldRetractionKernal(T* news, T* olds, T* grad, size_s n, double lr){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        news[i] = olds[i] * exp(lr*grad[i]/olds[i]);
    }  
}

template <typename T>
__global__ void ObjectiveLambdaKernal(T* news, T* olds, size_s n){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        news[i] = (olds[i] * olds[i] - 1)*(olds[i] * olds[i] - 1);
    }  
}

template <typename T>
__global__ void GradLambdaKernal(T* news, T* olds, size_s n){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        news[i] = (olds[i+1] * olds[i+1] - 1)*olds[i+1];
    }  
}

template <typename T>
__global__ void HessLambdaKernal(T* news, T* olds, T* oldsu, size_s n){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        news[i] = (3 * olds[i+1] * olds[i+1] - 1) * oldsu[i+1];
    }  
}

// template <typename T>
// __global__ void compute_sum(T* v_s, T* s0, double* result, size_s size) {
//     size_s idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         double temp = (v_s[idx] * v_s[idx]) / (s0[idx] * s0[idx]);
//         atomicAdd(result, temp);
//     }
// }

// template <typename T>
// __global__ void ProductManifoldInnerKernal(T* As, T* Bs, T* s0, double* result, size_s n){
//     size_s i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i<n){
//         result
//     }  
// }

double ProductManifoldInner(DeviceBlasHandle& CUOPT_blas_handle, opt_var &AR,opt_var &BR, opt_var &As, opt_var &Bs){
    double result1 = 0;
    double result2 = 0;
    CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, AR.total_size, AR.vals, 1, BR.vals, 1, &result1));
    CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, As.total_size, As.vals, 1, Bs.vals, 1, &result2));
    
    return result1+result2;
}


void XMtrustregion(opt_var &C, const opt_var &R0, const opt_var &s0, opt_var &R_result, opt_var &s_result,const double lam, double &gradtol, double linesearch_step , opt_var &v , double* primal_value, const double maxtime){
    // initialize
    DeviceBlasHandle CUOPT_blas_handle;
    CUOPT_blas_handle.activate();
    // DeviceSolverDnHandle CUOPT_cuslover_handle;
    // CUOPT_cuslover_handle.activate();
    
    // size_s n = 5;
    // std::vector<datatype> Q_h(9*n*n);
    // for(size_l i = 0; i<9*n*n; ++i){
    //     Q_h[i] = static_cast<float>(rand()) / RAND_MAX;
    // }
    // opt_var A({3*n,3*n});
    // A.SynchronizeHostToDevice(Q_h.data());
    // opt_var C({3*n,3*n});
    // DnMatDnMat(CUOPT_blas_handle,C,A,A,CUBLAS_OP_N,CUBLAS_OP_T);

    // size_s n = 0;
    // std::vector<datatype> Q_h;
    // loadMatrixFromBin("../assets/matrix_data_20.bin", Q_h, n); 
    // opt_var C({3*n,3*n});
    // C.SynchronizeHostToDevice(Q_h.data());

    

    size_s o = R0.dimensions[1];
    size_s n = C.dimensions[0]/3;
    size_s dim =  n * (3 * o - 6) + n - 1;
    double delta_bar = sqrt(dim);
    double delta = delta_bar / 8.0;

    // copy R
    opt_var R({3*n,o});
    CHECK_CUDA(cudaMemcpy(R.vals, R0.vals, R.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));
    
    // opt_var R({3*n,o});
    // // generate r0
    // std::vector<datatype> R_h(3*o*n,0);
    // for(size_l i = 0; i<n; ++i){
    //     R_h[3*i] = 1.0;
    //     R_h[3*i+3*n+1] = 1.0;
    //     R_h[3*i+6*n+2] = 1.0;
    // }
     
    // R.SynchronizeHostToDevice(R_h.data());
    //s = ones(n-1,1);
    //s_ex = [1,s];
    // SUPERRRRRRRRR UGLY!!!!!!!!!!
    opt_var s_ex({n}); //because we reserve the first element to be 1
    std::vector<datatype> s_ex_h(n,1);
    s_ex.SynchronizeHostToDevice(s_ex_h.data());
    opt_var s; //because we reserve the first element to be 1
    s.vals = s_ex.vals+1;
    s.num_dims = 1;
    s.dimensions = new size_s[1];
    s.dimensions[0] = n-1;
    s.total_size = n-1;
    // copy s
    CHECK_CUDA(cudaMemcpy(s.vals, s0.vals, s.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));

    opt_var p_s_ex({n}); //because we reserve the first element to be 1
    std::vector<datatype> p_s_ex_h(n,1);
    p_s_ex.SynchronizeHostToDevice(p_s_ex_h.data());
    p_s_ex.setValueAt(0,0);
    opt_var p_s; //because we reserve the first element to be 1
    p_s.vals = p_s_ex.vals+1;
    p_s.num_dims = 1;
    p_s.dimensions = new size_s[1];
    p_s.dimensions[0] = n-1;
    p_s.total_size = n-1;

    

    opt_var sR({3*n,o});
    opt_var dfdsR({3*n,o});
    dnmat_mul_spdiag_batch(sR,R,s_ex,3);

    opt_var grad_r({3*n,o});
    opt_var grad_s_pre({3*n,o});
    opt_var grad_s({n-1});
    opt_var slamobj({n-1});
    opt_var grad_s_ex({n});
    opt_var CsR({3*n,o});

    // the cost is sR * C * sR^T
    auto objc = [&C,&CUOPT_blas_handle, &CsR, &n, &slamobj,&lam](opt_var& sR_point_T,opt_var& s_point) {
        double result = 0;
        double result_lambda = 0;
        DnMatDnMat(CUOPT_blas_handle,CsR,C,sR_point_T); 
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, sR_point_T.total_size, CsR.vals, 1, sR_point_T.vals, 1, &result));
        ObjectiveLambdaKernal<<<(n + 1024 - 2) / 1024,1024>>>(slamobj.vals,s_point.vals,n-1);   
        CHECK_CUBLAS(cublasDasum(CUOPT_blas_handle.cublas_handle, slamobj.total_size, slamobj.vals, 1, &result_lambda));
        return result + lam * result_lambda;
    };
    

    //function [gradR,grads] = grad(AAT,R,s)
    //    n = size(R,1)/3;
    //    s_ex = [1;s];
    //    sR = R.*kron(s_ex,ones(3,1));
    //    dfdsR = 2 * AAT * sR;
    //    gradR = dfdsR.*kron(s_ex,ones(3,1));
    //    grads_pre = dfdsR.*R;
    //    grads = zeros(n-1,1);
    //    for i = 1:n-1
    //        grads(i) = sum(grads_pre(3*i+1:3*i+3,:),'all');
    //    end
    // end
    opt_var slamgrad({n-1});
    auto grad = [&C,&CUOPT_blas_handle,&grad_r,&grad_s,&dfdsR,&n,&slamgrad,&lam](opt_var& R_point, opt_var& s_point, opt_var& sR_point){
        DnMatDnMat(CUOPT_blas_handle,dfdsR,C,sR_point,CUBLAS_OP_N,CUBLAS_OP_N,2.0,0.0); 
        dnmat_mul_spdiag_batch(grad_r,dfdsR,s_point,3);
        dnmat_Ddot_colomn_batch(grad_s,dfdsR,R_point,3);
        GradLambdaKernal<<<(n + 1024 - 2) / 1024,1024>>>(slamgrad.vals,s_point.vals,n-1);   
        double glam = 4 * lam;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, slamgrad.total_size, &glam, slamgrad.vals, 1, grad_s.vals, 1));
        return;
    };


    // function [hR,hs] = hess(AAT,R,s,Ru,su)
    //     % Hank: can you check if my gradient is correct?
    //     n = size(R,1)/3;
    //     o = size(R,2);
    //     s_ex = [1;s]; % add first scale = 1
    //     s_ex = kron(s_ex,ones(3,1));
    //     sR = R.*s_ex;  
    //     su_ex = [0;su]; % add first scale = 1
    //     su_ex = kron(su_ex,ones(3,1));   
    //     sRu = Ru .* s_ex;
    //     suR = R .* su_ex;
    //     hR = (2*AAT*(sRu+suR)).*s_ex+(2*AAT*sR).*su_ex;
    //     hs_pre = (2*AAT*(sRu+suR)).*R + (2*AAT*sR).*Ru;
    //     hs = zeros(n-1,1);
    //     for i = 1:n-1
    //        hs(i) = sum(hs_pre(3*i+1:3*i+3,:),'all');
    //     end
    // end
    opt_var hr({3*n,o});
    opt_var hrT({o,3*n});
    opt_var hs({n-1});
    opt_var slamhess({n-1});
    opt_var sRu({3*n,o});
    opt_var suR({3*n,o});
    opt_var CsRu({3*n,o});
    opt_var suCsR({3*n,o});
    opt_var sCsRu({3*n,o});
    opt_var CsRudotR({n-1});
    opt_var CsRdotRu({n-1});
    // input R^T
    auto ehess = [&C,&CUOPT_blas_handle,&hr,&hs,&sR,&sRu,&suR,&CsR,&CsRu,&suCsR,&sCsRu,&CsRudotR,&CsRdotRu,&slamhess,&n,&lam](opt_var& R_point, opt_var& s_point, opt_var& Ru_point, opt_var& su_point){
        //dnmat_mul_spdiag_batch(sR,R_point,s_point,3);
        dnmat_mul_spdiag_batch(sRu,Ru_point,s_point,3);
        dnmat_mul_spdiag_batch(suR,R_point,su_point,3);

        //sRu = sRu + suR;
        double alpha = 1.0;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, sRu.total_size, &alpha, suR.vals, 1, sRu.vals, 1));
        
        //DnMatDnMat(CUOPT_blas_handle,CsR,C,sR,CUBLAS_OP_N,CUBLAS_OP_N,2.0,0.0); 
        DnMatDnMat(CUOPT_blas_handle,CsRu,C,sRu,CUBLAS_OP_N,CUBLAS_OP_N,2.0,0.0);
        dnmat_mul_spdiag_batch(suCsR,CsR,su_point,3);
        dnmat_mul_spdiag_batch(sCsRu,CsRu,s_point,3);
        // hr = suCsR + sCsRu;
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, hr.total_size, sCsRu.vals, 1, hr.vals, 1));
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hr.total_size, &alpha, suCsR.vals, 1, hr.vals, 1));

        dnmat_Ddot_colomn_batch(CsRudotR,CsRu,R_point,3);
        dnmat_Ddot_colomn_batch(CsRdotRu,CsR,Ru_point,3);
        //hs = CsRudotR + CsRdotRu;
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, hs.total_size, CsRudotR.vals, 1, hs.vals, 1));
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hs.total_size, &alpha, CsRdotRu.vals, 1, hs.vals, 1));

        //hs = hs + 4*lam*su.*(3*s.^2-1);
        
        HessLambdaKernal<<<(n + 1024 - 2) / 1024,1024>>>(slamhess.vals,s_point.vals,su_point.vals,n-1); 
        double hlam = 4 * lam;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, slamhess.total_size, &hlam, slamhess.vals, 1, hs.vals, 1));
    };


    // function [rhR,rhs] = ehess2rhess(ehessR,ehesss,egradR,egrads,R,s,Ru,su)
    //     n = size(egradR,1)/3;
    //     rhR = zeros(size(egradR));
    //     for i = 1:n
    //         rhR(3*i-2:3*i,:) = ehessR(3*i-2:3*i,:) - Ru(3*i-2:3*i,:)*sym(R(3*i-2:3*i,:)'*egradR(3*i-2:3*i,:));
    //         rhR(3*i-2:3*i,:) = rhR(3*i-2:3*i,:) - R(3*i-2:3*i,:)*sym(R(3*i-2:3*i,:)'*rhR(3*i-2:3*i,:));
    //     end
    //     rhs = ehesss .* (s.^2) - su .* egrads .* s;
    // end

    // input R
    opt_var RTgradR({3,3*n});
    opt_var RTgradR_sym({3,3*n});   
    opt_var rhr({o,3*n});
    opt_var rhs({n-1});
    opt_var RTrhr({3,3*n});
    opt_var RTrhr_sym({3,3*n}); 
    opt_var sus({n-1});
    opt_var suhss({n-1});
    auto ehess2rhess = [&CUOPT_blas_handle,&RTgradR,&RTgradR_sym,&rhr,&rhs,&RTrhr,&RTrhr_sym,&sus,&suhss](opt_var& ehessR, opt_var& ehesss, opt_var& egradR, opt_var& egrads, opt_var& R_point, opt_var& s_point, opt_var& Ru_point, opt_var& su_point){
        size_s n = R_point.dimensions[1]/3;
        DnMatDnMatBatch(CUOPT_blas_handle,RTgradR,R_point,egradR,3,R_point.dimensions[0],3,n,CUBLAS_OP_T,CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals,RTgradR.vals,RTgradR.dimensions[0],n);
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, ehessR.total_size, ehessR.vals, 1, rhr.vals, 1));
        CHECK_CUDA(cudaDeviceSynchronize());
        DnMatDnMatBatch(CUOPT_blas_handle,rhr,Ru_point,RTgradR_sym,Ru_point.dimensions[0],3,3,n,CUBLAS_OP_N,CUBLAS_OP_N,-1.0,1.0);
        
        DnMatDnMatBatch(CUOPT_blas_handle,RTrhr,R_point,rhr,3,R_point.dimensions[0],3,n,CUBLAS_OP_T,CUBLAS_OP_N);
        symBatched(RTrhr_sym.vals,RTrhr.vals,RTrhr.dimensions[0],n);
        DnMatDnMatBatch(CUOPT_blas_handle,rhr,R_point,RTrhr_sym,R_point.dimensions[0],3,3,n,CUBLAS_OP_N,CUBLAS_OP_N,-1.0,1.0);
        DnMatDnMatDot(rhs,ehesss,s_point,2);
        DnMatDnMatDot(sus,su_point,s_point,1);
        DnMatDnMatDot(suhss,sus,egrads,1);
        // rhs = rhs - suhss;
        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, rhs.total_size, &one, suhss.vals, 1, rhs.vals, 1));
        return;
    };

    // function [rgradR,rgrads] = projection(egradR,egrads,R,s)
    //     n = size(egradR,1)/3;
    //     rgradR = zeros(size(egradR));
    //     for i = 1:n
    //         rgradR(3*i-2:3*i,:) = egradR(3*i-2:3*i,:) - R(3*i-2:3*i,:)*sym(R(3*i-2:3*i,:)'*egradR(3*i-2:3*i,:));
    //     end
    //     rgrads = (s.^2).*egrads;
    // end
    opt_var rgrad_r({o,3*n});
    opt_var rgrad_s({n-1});
    auto projection = [&C,&CUOPT_blas_handle,&RTgradR,&RTgradR_sym,&rgrad_r,&rgrad_s](opt_var& R_point, opt_var& s_point, opt_var& R_gradient, opt_var& s_gradient){
        //round on n
        size_s n = R_point.dimensions[1]/3;
        DnMatDnMatBatch(CUOPT_blas_handle,RTgradR,R_point,R_gradient,3,R_point.dimensions[0],3,n,CUBLAS_OP_T,CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals,RTgradR.vals,RTgradR.dimensions[0],n);
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R_gradient.total_size, R_gradient.vals, 1, rgrad_r.vals, 1));
        CHECK_CUDA(cudaDeviceSynchronize());
        DnMatDnMatBatch(CUOPT_blas_handle,rgrad_r,R_point,RTgradR_sym,R_point.dimensions[0],3,3,n,CUBLAS_OP_N,CUBLAS_OP_N,-1.0,1.0);
        DnMatDnMatDot(rgrad_s,s_gradient,s_point,2);
        return;
    };

    //function [newR,news] = retraction(rgradR,rgrads,R,s,lr)
    //     n = size(R,1)/3;
    //     for i = 1:n
    //         [q,r] = qr(R(3*i-2:3*i,:)-lr*rgradR(3*i-2:3*i,:));
    //         sig = sign(diag(r));
    //         newR(3*i-2:3*i,:) = q.*sig';
    //     end
    //     news = s.*exp(-lr*rgrads./s);
    // end
    opt_var new_R_T({o,3*n});
    opt_var new_R_T_test({o,3});
    opt_var new_R({3*n,o});
    opt_var new_s_ex({n}); //because we reserve the first element to be 1
    std::vector<datatype> new_s_ex_h(n,1);
    new_s_ex.SynchronizeHostToDevice(new_s_ex_h.data());
    opt_var new_s; //because we reserve the first element to be 1
    new_s.vals = new_s_ex.vals+1;
    new_s.num_dims = 1;
    new_s.dimensions = new size_s[1];
    new_s.dimensions[0] = n-1;
    new_s.total_size = n-1;

    auto retraction = [&CUOPT_blas_handle,&new_R_T,&new_s](opt_var& R_point, opt_var& s_point, opt_var& rR_gradient, opt_var& rs_gradient, double lr){
        // annoying thing is qr cannot explicitly return q and r
        // we can calculate q ourselves
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, R_point.total_size, &lr, rR_gradient.vals, 1, R_point.vals, 1));

        batchedQR(new_R_T,R_point,3,s_point.total_size+1);
        //batchedQR(new_R_T_test,R_point,3,1);
        //new_R_T.print();
        positiveManifoldRetractionKernal<<<(s_point.total_size + 1024 - 1) / 1024,1024>>>(new_s.vals,s_point.vals,rs_gradient.vals,s_point.total_size,lr);
        return;
    };

    opt_var R_T({o,3*n});
    transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
    opt_var grad_r_T({o,3*n});  


    opt_var sR_new({3*n,o});
    std::cout<<"start linesearch"<<std::endl;
    if(linesearch_step != 0){
        // do linesearch here
        double f0 = objc(sR,s);
        double alpha = linesearch_step;
        // decent R is from input v
        opt_var decent_dir_R({3*n,o});
        CHECK_CUDA(cudaMemcpyAsync(decent_dir_R.vals + 3*n*(o-1), v.vals, sizeof(datatype) * v.total_size, cudaMemcpyDeviceToDevice));
        opt_var decent_dir_R_T({o,3*n});
        transpose(decent_dir_R_T.vals,decent_dir_R.vals,decent_dir_R.dimensions[0],decent_dir_R.dimensions[1]);
        // decent s is 0
        opt_var decent_dir_s({n-1});

        retraction(R_T,s,decent_dir_R_T,decent_dir_s,-alpha);
        transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
        transpose(new_R.vals,new_R_T.vals,new_R_T.dimensions[0],new_R_T.dimensions[1]);
        dnmat_mul_spdiag_batch(sR_new,new_R,s_ex,3);
        double f = objc(sR_new,s);
        while(f>f0){
            alpha = alpha/2;
            retraction(R_T,s,decent_dir_R_T,decent_dir_s,-alpha);
            transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
            transpose(new_R.vals,new_R_T.vals,new_R_T.dimensions[0],new_R_T.dimensions[1]);
            dnmat_mul_spdiag_batch(sR_new,new_R,s_ex,3);
            f = objc(sR_new,s);
            if(alpha<1e-20){
                printf("linesearch failed! BM stopped! \n");
                s.vals = nullptr;
                p_s.vals = nullptr;
                new_s.vals = nullptr;
                *primal_value = -1;
                return;
            }
        }

        if(f0-f>0){
            printf("linesearch decrease %1.3e\n",f0-f);
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R_T.total_size, new_R_T.vals, 1, R_T.vals, 1));
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R.total_size, new_R.vals, 1, R.vals, 1));
        }
        else{
            printf("linesearch failed! BM stopped! \n");
            s.vals = nullptr;
            p_s.vals = nullptr;
            new_s.vals = nullptr;
            *primal_value = -1;
            return;
        }

    }


    // opt_var sR_T({o,3*n});
    // opt_var sR_TsR({o,o});
    // transpose(sR_T.vals,sR.vals,sR.dimensions[0],sR.dimensions[1]);
    // DnMatDnMat(CUOPT_blas_handle,sR_TsR,sR_T,sR); 
    // sR_TsR.print();
    size_s max_inner_iter = 1000;
    size_s max_outer_iter = 1000;
    size_s totalite = 0;
  
    double* loss = new double[max_outer_iter];
    double* gradnorm = new double[max_outer_iter];
    loss[0] = objc(sR,s);
    
    // 1 for nagative curvature
    // 2 for exceed trust region
    // 3 for reached norm tolerance
    // 5 for numerical issue
    // 6 for max iteration
    int endreason = 6;
    // 1 for tr region shrink
    // 2 for tr region expand
    // 3 for tr region REJ
    // 4 for tr region remain
    int trstatus = 4;
    int shrink_count = 0;

    opt_var rhsds({n-1});
    opt_var rsds({n-1});
    opt_var psds({n-1});
    opt_var vsds({n-1});
    opt_var hvsds({n-1});
    opt_var rgradsds({n-1});

    opt_var r_R({o,3*n});
    opt_var r_s({n-1});
    opt_var p_R({o,3*n});
    opt_var p_R_T({3*n,o});
    opt_var v_R_T({3*n,o});
    size_s i = 0;
    size_s k = 0;
    auto start = high_resolution_clock::now();
    for(k = 0; k<max_outer_iter;++k){

        opt_var v_R({o,3*n});
        opt_var v_s({n-1});

        opt_var hv_R({o,3*n});
        opt_var hv_s({n-1});
        opt_var bestR(R);
        opt_var bests(s);
        double bestloss = loss[k];
        double* rdotr = new double[max_inner_iter];
        double* lossqu = new double[max_inner_iter];
        lossqu[0] = 0;
        double* lossqu1 = new double[max_inner_iter];
        lossqu1[0] = 0;
        grad(R,s_ex,sR);
        transpose(grad_r_T.vals,grad_r.vals,grad_r.dimensions[0],grad_r.dimensions[1]);
        transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
        projection(R_T,s,grad_r_T,grad_s);
        DnMatDnMatDivide(rgradsds,rgrad_s,s,1);
        // // r_R = rgradR;
        // // r_s = rgrads;
        // // p_R = -r_R;
        // // p_s = -r_s;
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, r_R.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, rgrad_s.vals, 1, r_s.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, p_R.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, rgrad_s.vals, 1, p_s.vals, 1));
        double neg_one = -1.0;
        cublasDscal(CUOPT_blas_handle.cublas_handle, p_R.total_size, &neg_one, p_R.vals, 1);
        cublasDscal(CUOPT_blas_handle.cublas_handle, p_s.total_size, &neg_one, p_s.vals, 1);
        DnMatDnMatDivide(rsds,r_s,s,1);
        rdotr[0] = ProductManifoldInner(CUOPT_blas_handle,r_R,r_R,rsds,rsds);
        gradnorm[k] = sqrt(rdotr[0]);
        
        if(k>0){   
        switch (trstatus)
        {
        case 1:
            std::cout << "TR- " ;
            break;
        case 2: 
            std::cout << "TR+ ";
            break;
        case 3:
            std::cout << "REJ " ;
            break;
        case 4:
            std::cout << "TR " ;
            break;
        }
        }
        printf("%d   %d   %1.3e   %1.3e",k, i+1,loss[k],gradnorm[k]);
        if(k>0){   
        switch (endreason)
        {
        case 1:
            std::cout << "   nagative curvature" << std::endl;
            break;
        case 2: 
            std::cout << "   exceed trust region" << std::endl;
            break;
        case 3:
            std::cout << "   reached norm tolerance" << std::endl;
            break;
        case 5:
            std::cout << "   numerical issue" << std::endl;
            break;
        case 6:
            std::cout << "   max iteration" << std::endl;
            break;
        }
        }else{
            printf("\n");
        }
        if(endreason == 5){
            printf("Terminate because of rdotr touched machine precise\n");
            break;
        }

        if(gradnorm[k] < gradtol){
            printf("Terminate because of small gradient norm\n");
            gradtol /= 10;
            break;
        }

        auto middle_end_time = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(middle_end_time - start);
        if(duration.count() > maxtime){
            printf("Terminate because of time limit\n");
            break;
        }
        endreason = 6;
        trstatus = 4;
        double* vdotv_record = new double[max_inner_iter];
        double* vdotp_record = new double[max_inner_iter];
        double* pdotp_record = new double[max_inner_iter];
        vdotv_record[0] = 0;
        vdotp_record[0] = 0;
        pdotp_record[0] = rdotr[0];
        endreason = 6;
        DnMatDnMat(CUOPT_blas_handle,CsR,C,sR,CUBLAS_OP_N,CUBLAS_OP_N,2.0,0.0); 
        CHECK_CUDA(cudaDeviceSynchronize());

        // double violation = 0;
        // double another_alpha = 0;

        for(i = 0; i<max_inner_iter;++i){
            transpose(p_R_T.vals,p_R.vals,p_R.dimensions[0],p_R.dimensions[1]);
            ehess(R,s_ex,p_R_T,p_s_ex);
            transpose(hrT.vals,hr.vals,hr.dimensions[0],hr.dimensions[1]);
            ehess2rhess(hrT,hs,grad_r_T,grad_s,R_T,s,p_R,p_s);
            //calculate inner
            DnMatDnMatDivide(rhsds,rhs,s,2);
            double alpha = rdotr[i] / ProductManifoldInner(CUOPT_blas_handle,p_R,rhr,p_s,rhsds);
            
            double vdotv = vdotv_record[i];
            double pdotp = pdotp_record[i];
            double vdotp = vdotp_record[i];
            
            if (rdotr[i] < 1e-15) {
                printf("gradient residual is very small!\n");
                endreason = 5; 
                break;
            }           
            if(alpha <=0){
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (-vdotp + sqrt_val) / pdotp;
                // v_R = v_R + tau * p_R;
                // v_s = v_s + tau * p_s;
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &tau, p_s.vals, 1, v_s.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &tau, rhr.vals, 1, hv_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &tau, rhs.vals, 1, hv_s.vals, 1));
                endreason = 1;
                break;
            }
            if(vdotv + 2*alpha*vdotp + alpha*alpha*pdotp > delta * delta){
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (-vdotp + sqrt_val) / pdotp;
                // v_R = v_R + tau * p_R;
                // v_s = v_s + tau * p_s;
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &tau, p_s.vals, 1, v_s.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &tau, rhr.vals, 1, hv_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &tau, rhs.vals, 1, hv_s.vals, 1));
                endreason = 2;
                break;
            }
            // v_R = v_R + alpha * p_R;
            // r_R = r_R + alpha * rhR;
            // v_s = v_s + alpha * p_s;
            // r_s = r_s + alpha * rhs;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &alpha, p_R.vals, 1, v_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &alpha, p_s.vals, 1, v_s.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, r_R.total_size, &alpha, rhr.vals, 1, r_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, r_s.total_size, &alpha, rhs.vals, 1, r_s.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &alpha, rhr.vals, 1, hv_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &alpha, rhs.vals, 1, hv_s.vals, 1));

            // retraction(R_T,s,v_R,v_s,1);
            // transpose(new_R.vals,new_R_T.vals,new_R_T.dimensions[0],new_R_T.dimensions[1]);
            // dnmat_mul_spdiag_batch(sR,new_R,new_s_ex,3);
            // double lossgt = objc(sR);
            // if(lossgt < bestloss){
            //     CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R.total_size, new_R.vals, 1, bestR.vals, 1));
            //     CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_s.total_size, new_s.vals, 1, bests.vals, 1));
                
            //     bestloss = lossgt;
            // }
            // // recover RT
            // transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
            
            DnMatDnMatDivide(rsds,r_s,s,1);
            rdotr[i+1] = ProductManifoldInner(CUOPT_blas_handle,r_R,r_R,rsds,rsds);
            if(sqrt(rdotr[i+1]) < gradnorm[k] * min(gradnorm[k],0.1)){
                endreason = 3;
                break;
            }
            // beta = rdotr(end)/rdotr(end-1);
            // p_R = -r_R + beta * p_R;
            // p_s = -r_s + beta * p_s;
            double beta = rdotr[i+1] / rdotr[i];
            CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_R.total_size, &beta, p_R.vals, 1));
            CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_s.total_size, &beta, p_s.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, p_R.total_size, &neg_one, r_R.vals, 1, p_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, p_s.total_size, &neg_one, r_s.vals, 1, p_s.vals, 1));
            // vdotv_record(end+1) = vdotv + 2*alpha*vdotp + alpha^2*pdotp;
            // vdotp_record(end+1) = beta*(vdotp+alpha*pdotp);
            // pdotp_record(end+1) = beta^2*pdotp + rdotr(end);
            vdotv_record[i+1] = vdotv + 2*alpha*vdotp + alpha*alpha*pdotp;
            vdotp_record[i+1] = beta*(vdotp+alpha*pdotp);
            pdotp_record[i+1] = beta*beta*pdotp + rdotr[i+1];

            // DnMatDnMatDivide(vsds,v_s,s,1);
            // DnMatDnMatDivide(hvsds,hv_s,s,1);
            // DnMatDnMatDivide(rgradsds,rgrad_s,s,1);
            // lossqu[i+1] = ProductManifoldInner(CUOPT_blas_handle,v_R,hv_R,vsds,hvsds)/2 + ProductManifoldInner(CUOPT_blas_handle,v_R,rgrad_r,vsds,rgradsds);
            // lossqu1[i+1] = ProductManifoldInner(CUOPT_blas_handle,v_R,hv_R,vsds,hvsds);
            // if(lossqu[i+1] >= lossqu[i]){
            //     s.print();
            //     printf("error! loss_qu is increasing\n");
            // }
            // DnMatDnMatDivide(psds,p_s,s,1);
            // DnMatDnMatDivide(rsds,r_s,s,1);
            // violation = ProductManifoldInner(CUOPT_blas_handle,p_R,r_R,psds,rsds)/sqrt(pdotp_record[i+1]*rdotr[i+1]);
            // another_alpha = ProductManifoldInner(CUOPT_blas_handle,p_R,r_R,psds,rsds);
            // opt_var another_r(rgrad_r);
            // double one = 1.0;
            // CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, another_r.total_size, &one, hv_R.vals, 1, another_r.vals, 1));
            // CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, another_r.total_size, &neg_one, r_R.vals, 1, another_r.vals, 1));
            
        }
        
        totalite += i + 1;
        DnMatDnMatDivide(vsds,v_s,s,2);
        double loss_qu = ProductManifoldInner(CUOPT_blas_handle,v_R,hv_R,vsds,hv_s)/2 + ProductManifoldInner(CUOPT_blas_handle,v_R,rgrad_r,vsds,rgrad_s);
        if(loss_qu >= 0){
            printf("error! loss_qu is larger than 0\n");
            break;
        }
        retraction(R_T,s,v_R,v_s,1);
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R_T.total_size, new_R_T.vals, 1, R_T.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_s.total_size, new_s.vals, 1, s.vals, 1));
        transpose(R.vals,R_T.vals,R_T.dimensions[0],R_T.dimensions[1]);
        dnmat_mul_spdiag_batch(sR,R,s_ex,3);
        loss[k+1] = objc(sR,s);

        double rou = (loss[k+1] - loss[k]) / loss_qu;
        //std::cout << "rou: " << rou ;
        if(rou < 0.25){
            delta = delta * 0.25;
            trstatus = 1;
            shrink_count ++;
        }else if(rou > 0.75 && endreason <= 2){
            delta = min(delta * 2,delta_bar);
            trstatus = 2;
            shrink_count = 0;
        }else{
            shrink_count = 0;
        }
        if(shrink_count > 3){
            delta = delta * 1e-3;
            shrink_count = 0;
            printf("delta shrinked to %1.3e\n",delta);
            if(delta < 1e-20){
                printf("delta is too small, BM stopped!\n");
                break;
            }
        }
        if(loss[k+1] > bestloss || rou < 0.1){
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R.total_size, bestR.vals, 1, R.vals, 1));
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, s.total_size, bests.vals, 1, s.vals, 1));
            loss[k+1] = bestloss;
            dnmat_mul_spdiag_batch(sR,R,s_ex,3);
            trstatus = 3;
        }
        
    }
    std::cout <<std::endl<< "Total iteration:     " << totalite <<std::endl;
    auto end = high_resolution_clock::now();
        std::cout << "Time taken by function1: "
            << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;
    *primal_value = loss[k];
    // record result
    CHECK_CUDA(cudaMemcpy(R_result.vals,R.vals,R.total_size * sizeof(datatype),cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(s_result.vals,s.vals,s.total_size * sizeof(datatype),cudaMemcpyDeviceToDevice));

    // must equal null, or there will be a memory leak
    s.vals = nullptr;
    p_s.vals = nullptr;
    new_s.vals = nullptr;
}