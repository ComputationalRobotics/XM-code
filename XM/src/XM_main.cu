#include <iostream>
#include <XM/trustregion.h>
#include <XM/checkeig.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
__global__ void DecentDirectionKernal(T* v, T* s, size_s n){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        v[3*i] /= s[i] ;
        v[3*i+1] /= s[i] ;
        v[3*i+2] /= s[i] ;
    }  
}

void loadCMatrixFromBin(const std::string& filename, std::vector<double>& matrix, size_s& n) {
    // Load the matrix from a binary file
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "cannot open file" << std::endl;
        return;
    }
    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    n = rows/3;
    matrix.resize(rows * cols);
    file.read(reinterpret_cast<char*>(matrix.data()), sizeof(double) * rows * cols);
    printf("rows: %d, cols: %d\n", rows, cols);
    file.close();
}

int solve_rebuttle(const std::string& dataset_path, size_s max_rank = 10, double tol = 1e-6, double lam = 0.0, double max_time = 1000) {
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Begin XM" << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

    std::string Q_matrix_filename = dataset_path + "/Q.bin";
    std::string R_matrix_filename = dataset_path + "/R_ini.bin";
    std::string s_matrix_filename = dataset_path + "/s_ini.bin";
    std::string output_path = dataset_path + "/";
    size_s n = 0;
    // load Q matrix on host
    std::vector<datatype> Q_h;
    loadCMatrixFromBin(Q_matrix_filename, Q_h, n); 
    opt_var C({3*n,3*n});
    C.SynchronizeHostToDevice(Q_h.data());

    // init
    // order of R is 3*n*o
    size_s o = 3;
    opt_var v({3*n});
    opt_var R0({3*n,o});
    
    // can be replaced by given initial guess
    std::vector<datatype> s0_ex_h(n,1);
    std::vector<datatype> R0_h(3*o*n,0);

    size_s n_RE = 0;
    loadCMatrixFromBin(R_matrix_filename, R0_h, n_RE);
    loadCMatrixFromBin(s_matrix_filename, s0_ex_h, n_RE);
    R0.SynchronizeHostToDevice(R0_h.data());

    opt_var s0_ex({n}); 
    s0_ex.SynchronizeHostToDevice(s0_ex_h.data());
    opt_var s0; //because we reserve the first element to be 1
    s0.vals = s0_ex.vals+1;
    s0.num_dims = 1;
    s0.dimensions = new size_s[1];
    s0.dimensions[0] = n-1;
    s0.total_size = n-1;

    opt_var s_ex({n}); 
    s_ex.SynchronizeHostToDevice(s0_ex_h.data());
    opt_var s; //because we reserve the first element to be 1
    s.vals = s_ex.vals+1;
    s.num_dims = 1;
    s.dimensions = new size_s[1];
    s.dimensions[0] = n-1;
    s.total_size = n-1;

    double gradtol = tol;
    double primal = 0;

    int status = 0;
    while(o<=max_rank){
        
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Solve TR with Rank   " << o <<std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

        opt_var R({3*n,o});
        if(o==3){
            std::vector<datatype> R0_h(3*o*n,0);
            for(size_l i = 0; i<n; ++i){
                R0_h[3*i] = 1.0;
                R0_h[3*i+3*n+1] = 1.0;
                R0_h[3*i+6*n+2] = 1.0;
            }
            R0.SynchronizeHostToDevice(R0_h.data());
            XMtrustregion(C,R0,s0,R,s,lam,gradtol,0,v,&primal, max_time);
        }
        else{
            XMtrustregion(C,R0,s0,R,s,lam,gradtol,1,v,&primal, max_time);
        }

        if(primal < 0){
            status = -2;
            o += 1;
            break;
        }
        

        // deal with constrain 
        opt_var sR({3*n,o});

        dnmat_mul_spdiag_batch(sR,R,s_ex,3);
        sR.SynchronizeDevicetoHost();
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Check Eigen value" << std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        
        if(checkeig(C,sR,lam,v,primal)){
            o += 1;
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals, sizeof(datatype) * R.total_size, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals, sizeof(datatype) * s.total_size, cudaMemcpyDeviceToDevice));
            status = 1;
            break;
        }
        else if(o < max_rank){
            R0.dimensions[1] = o+1;
            R0.total_size = 3*n*(o+1);
            R0.allocate();
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals, sizeof(datatype) * R.total_size, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals, sizeof(datatype) * s.total_size, cudaMemcpyDeviceToDevice));
            DecentDirectionKernal<<<(3*n+255)/256, 256>>>(v.vals, s_ex.vals, n);
        }else{
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals, sizeof(datatype) * R.total_size, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals, sizeof(datatype) * s.total_size, cudaMemcpyDeviceToDevice));
            status = 2;
        }
        o += 1;
    }
    
    if(o > max_rank){
        std::cout << "BM stoped because max rank" << std::endl;
    }

    // save R and s
    std::vector<datatype> R_h(3*n*(o-1));
    CHECK_CUDA(cudaMemcpy(R_h.data(), R0.vals, sizeof(datatype) * R0.total_size, cudaMemcpyDeviceToHost));
    std::ofstream fileR(output_path + "R.bin", std::ios::binary);
    // write row and col
    int temp1 = 3 * n;
    fileR.write(reinterpret_cast<char*>(&temp1), sizeof(int));  
    int temp2 = o - 1;
    fileR.write(reinterpret_cast<char*>(&temp2), sizeof(int));

    fileR.write(reinterpret_cast<char*>(R_h.data()), sizeof(double) * 3 * n * (o-1));
    fileR.close();

    std::cout << "saved R" << std::endl;

    std::vector<datatype> s_h(n);
    CHECK_CUDA(cudaMemcpy(s_h.data(), s0_ex.vals, sizeof(datatype) * s0_ex.total_size, cudaMemcpyDeviceToHost));
    std::ofstream files(output_path + "s.bin", std::ios::binary);
    files.write(reinterpret_cast<char*>(&n), sizeof(int));
    int one = 1;
    files.write(reinterpret_cast<char*>(&one), sizeof(int));
    files.write(reinterpret_cast<char*>(s_h.data()), sizeof(double) * n);
    files.close();

    s0.vals = nullptr;
    s.vals = nullptr;
    return status;
}

void solve(const std::string& dataset_path, size_s max_rank = 10, double tol = 1e-6, double lam = 0.0, double max_time = 1000) {
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Begin XM" << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

    std::string Q_matrix_filename = dataset_path + "/Q.bin";
    std::string output_path = dataset_path + "/";
    size_s n = 0;
    // load Q matrix on host
    std::vector<datatype> Q_h;
    loadCMatrixFromBin(Q_matrix_filename, Q_h, n); 
    opt_var C({3*n,3*n});
    C.SynchronizeHostToDevice(Q_h.data());

    // init
    // order of R is 3*n*o
    size_s o = 3;
    opt_var v({3*n});
    opt_var R0({3*n,o});
    
    // can be replaced by given initial guess
    std::vector<datatype> s0_ex_h(n,1);

    opt_var s0_ex({n}); 
    s0_ex.SynchronizeHostToDevice(s0_ex_h.data());
    opt_var s0; //because we reserve the first element to be 1
    s0.vals = s0_ex.vals+1;
    s0.num_dims = 1;
    s0.dimensions = new size_s[1];
    s0.dimensions[0] = n-1;
    s0.total_size = n-1;

    opt_var s_ex({n}); 
    s_ex.SynchronizeHostToDevice(s0_ex_h.data());
    opt_var s; //because we reserve the first element to be 1
    s.vals = s_ex.vals+1;
    s.num_dims = 1;
    s.dimensions = new size_s[1];
    s.dimensions[0] = n-1;
    s.total_size = n-1;

    double gradtol = tol;
    double primal = 0;
    while(o<=max_rank){
        
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Solve TR with Rank   " << o <<std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

        opt_var R({3*n,o});
        if(o==3){
            std::vector<datatype> R0_h(3*o*n,0);
            for(size_l i = 0; i<n; ++i){
                R0_h[3*i] = 1.0;
                R0_h[3*i+3*n+1] = 1.0;
                R0_h[3*i+6*n+2] = 1.0;
            }
            R0.SynchronizeHostToDevice(R0_h.data());
            XMtrustregion(C,R0,s0,R,s,lam,gradtol,0,v,&primal, max_time);
        }
        else{
            XMtrustregion(C,R0,s0,R,s,lam,gradtol,1,v,&primal, max_time);
        }

        if(primal < 0){
            o += 1;
            break;
        }
        

        // deal with constrain 
        opt_var sR({3*n,o});

        dnmat_mul_spdiag_batch(sR,R,s_ex,3);
        sR.SynchronizeDevicetoHost();
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Check Eigen value" << std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        
        if(checkeig(C,sR,lam,v,primal)){
            o += 1;
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals, sizeof(datatype) * R.total_size, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals, sizeof(datatype) * s.total_size, cudaMemcpyDeviceToDevice));
            break;
        }
        else if(o < max_rank){
            R0.dimensions[1] = o+1;
            R0.total_size = 3*n*(o+1);
            R0.allocate();
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals, sizeof(datatype) * R.total_size, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals, sizeof(datatype) * s.total_size, cudaMemcpyDeviceToDevice));
            DecentDirectionKernal<<<(3*n+255)/256, 256>>>(v.vals, s_ex.vals, n);
        }else{
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals, sizeof(datatype) * R.total_size, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals, sizeof(datatype) * s.total_size, cudaMemcpyDeviceToDevice));
        }
        o += 1;
    }
    
    if(o > max_rank){
        std::cout << "BM stoped because max rank" << std::endl;
    }

    // save R and s
    std::vector<datatype> R_h(3*n*(o-1));
    CHECK_CUDA(cudaMemcpy(R_h.data(), R0.vals, sizeof(datatype) * R0.total_size, cudaMemcpyDeviceToHost));
    std::ofstream fileR(output_path + "R.bin", std::ios::binary);
    // write row and col
    int temp1 = 3 * n;
    fileR.write(reinterpret_cast<char*>(&temp1), sizeof(int));  
    int temp2 = o - 1;
    fileR.write(reinterpret_cast<char*>(&temp2), sizeof(int));

    fileR.write(reinterpret_cast<char*>(R_h.data()), sizeof(double) * 3 * n * (o-1));
    fileR.close();

    std::cout << "saved R" << std::endl;

    std::vector<datatype> s_h(n);
    CHECK_CUDA(cudaMemcpy(s_h.data(), s0_ex.vals, sizeof(datatype) * s0_ex.total_size, cudaMemcpyDeviceToHost));
    std::ofstream files(output_path + "s.bin", std::ios::binary);
    files.write(reinterpret_cast<char*>(&n), sizeof(int));
    int one = 1;
    files.write(reinterpret_cast<char*>(&one), sizeof(int));
    files.write(reinterpret_cast<char*>(s_h.data()), sizeof(double) * n);
    files.close();

    s0.vals = nullptr;
    s.vals = nullptr;
    return;
}

void solve_rank3(const std::string& dataset_path, size_s max_rank = 10, double tol = 1e-6, double lam = 0.0, double max_time = 1000) {
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Begin XM" << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

    std::string Q_matrix_filename = dataset_path + "/Q.bin";
    std::string output_path = dataset_path + "/";
    size_s n = 0;
    // load Q matrix on host
    std::vector<datatype> Q_h;
    loadCMatrixFromBin(Q_matrix_filename, Q_h, n); 
    opt_var C({3*n,3*n});
    C.SynchronizeHostToDevice(Q_h.data());

    // init
    // order of R is 3*n*o
    size_s o = 3;
    opt_var v({3*n});
    opt_var R0({3*n,o});
    
    // can be replaced by given initial guess
    std::vector<datatype> s0_ex_h(n,1);

    opt_var s0_ex({n}); 
    s0_ex.SynchronizeHostToDevice(s0_ex_h.data());
    opt_var s0; //because we reserve the first element to be 1
    s0.vals = s0_ex.vals+1;
    s0.num_dims = 1;
    s0.dimensions = new size_s[1];
    s0.dimensions[0] = n-1;
    s0.total_size = n-1;

    opt_var s_ex({n}); 
    s_ex.SynchronizeHostToDevice(s0_ex_h.data());
    opt_var s; //because we reserve the first element to be 1
    s.vals = s_ex.vals+1;
    s.num_dims = 1;
    s.dimensions = new size_s[1];
    s.dimensions[0] = n-1;
    s.total_size = n-1;

    double gradtol = tol;
    double primal = 0;

    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Solve TR with Rank   " << o <<std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

    opt_var R({3*n,o});

    std::vector<datatype> R0_h(3*o*n,0);
    for(size_l i = 0; i<n; ++i){
        R0_h[3*i] = 1.0;
        R0_h[3*i+3*n+1] = 1.0;
        R0_h[3*i+6*n+2] = 1.0;
    }
    R0.SynchronizeHostToDevice(R0_h.data());
    XMtrustregion(C,R0,s0,R,s,lam,gradtol,0,v,&primal, max_time);

    CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals, sizeof(datatype) * R.total_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals, sizeof(datatype) * s.total_size, cudaMemcpyDeviceToDevice));
    
    // save R and s
    std::vector<datatype> R_h(3*n*o);
    CHECK_CUDA(cudaMemcpy(R_h.data(), R0.vals, sizeof(datatype) * R0.total_size, cudaMemcpyDeviceToHost));
    std::ofstream fileR(output_path + "R.bin", std::ios::binary);
    // write row and col
    int temp1 = 3 * n;
    fileR.write(reinterpret_cast<char*>(&temp1), sizeof(int));  
    int temp2 = o;
    fileR.write(reinterpret_cast<char*>(&temp2), sizeof(int));

    fileR.write(reinterpret_cast<char*>(R_h.data()), sizeof(double) * 3 * n * o);
    fileR.close();

    std::cout << "saved R" << std::endl;

    std::vector<datatype> s_h(n);
    CHECK_CUDA(cudaMemcpy(s_h.data(), s0_ex.vals, sizeof(datatype) * s0_ex.total_size, cudaMemcpyDeviceToHost));
    std::ofstream files(output_path + "s.bin", std::ios::binary);
    files.write(reinterpret_cast<char*>(&n), sizeof(int));
    int one = 1;
    files.write(reinterpret_cast<char*>(&one), sizeof(int));
    files.write(reinterpret_cast<char*>(s_h.data()), sizeof(double) * n);
    files.close();

    s0.vals = nullptr;
    s.vals = nullptr;
    return;
}

PYBIND11_MODULE(XM, m) {
    m.doc() = "pybind11 for XM"; 
    m.def("solve", &solve, "XM main function");
    m.def("solve_rebuttle", &solve_rebuttle, "permit give initial guess");
    m.def("solve_rank3", &solve_rank3, "XM main function for rank 3 only");
}