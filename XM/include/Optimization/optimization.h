#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <iostream>
#include <functional>
#include <vector>
#include <Utils/memory.h>

using datatype = double;
using opt_objective = std::function<datatype(DeviceDnTen<datatype>&)>;
using opt_gradient = std::function<void(DeviceDnTen<datatype>&,DeviceDnTen<datatype>&)>;
using opt_var = DeviceDnTen<datatype>;

class opt_option {
    public:    
    enum Method {
        GradientDescent = 1,
        NewtonMethod = 2
    };

    opt_option(){};
    Method method = GradientDescent;

    // gradient decent
    double lr_gd = 0.1;
    size_l max_iteration_gd = 100;
};

class Optimizer {
public:
    Optimizer(opt_objective func, opt_gradient grad): F(func), gradF(grad){}
          
    void optimize(opt_var& initial_guess, opt_option& option,DeviceBlasHandle& cublas_H) {
        x.copy(initial_guess);
        gradx.copy(initial_guess);
        switch (option.method) {
            case opt_option::GradientDescent: {
                gradientdecent(option.lr_gd, option.max_iteration_gd,cublas_H);
                break;
            }
            case opt_option::NewtonMethod: {
                
                std::cout << "Newton Method is not implemented yet." << std::endl;
                break;
            }
        }

    }

    void gradientdecent(double lr, size_l maxitr,DeviceBlasHandle& cublas_H){
        std::cout << "Initial value is:" << F(x) << std::endl;
        double dlr = -lr;
        for(size_l i = 0; i<maxitr; ++i){
            gradF(x,gradx);
            CHECK_CUBLAS(cublasDaxpy(cublas_H.cublas_handle, x.total_size,
                           &dlr,gradx.vals, 1,x.vals, 1));   
            if(i % 10 == 0){
                std::cout << "New value is:" << F(x) << std::endl;
            }
        }
    }

// private:
    
    opt_objective F;
    opt_gradient gradF;
    datatype Fx;
    opt_var x;
    opt_var gradx;
};

#endif // OPTIMIZATION_H
