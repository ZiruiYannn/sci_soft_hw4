#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cassert>
#include <iostream>
#include "bfgs.hpp"
#include "io_siqrd.hpp"
#include "siqrd.hpp"

namespace ublas = boost::numeric::ublas;
typedef double prec;

int main(int argc, char* argv[]) {
    /*observation1*/
    int num_row, num_col;
    kul::observation_get_size("../observation/observations1.in", num_row, num_col);

    ublas::matrix<prec> obsv(num_row, num_col);
    
    prec T;
    
    kul::read_observation("../observation/observations1.in", obsv, T, num_row, num_col);

    ublas::vector<prec> x0(5);
    auto mr= ublas::row (obsv,0);
    x0.assign(mr);

    ublas::vector<prec> init_para(5);
    init_para(0) = 0.004; /*alpha*/
    init_para(1) = 0.32 ; /*beta*/
    init_para(2) = 0.151; /*gamma*/
    init_para(3) = 0.052; /*delta*/
    init_para(4) = 0.03; /*mu*/

    ublas::identity_matrix<prec> B0 (5);

    kul::bfgs try1(obsv, x0, T, num_row - 1, init_para, B0);
    auto result = try1.BFGS_heun();
    std::cout<<"result of observation1: "<<result<<std::endl;
    std::cout<<"LSE = "<< try1.lse_heun(result)<<std::endl;
    kul::write_prediction("../input/prediction1.out", result, x0);

    /*obseration2*/
    kul::observation_get_size("../observation/observations2.in", num_row, num_col);

    ublas::matrix<prec> obsv2(num_row, num_col);
    prec T2;
    
    kul::read_observation("../observation/observations2.in", obsv2, T2, num_row, num_col);

     ublas::vector<prec> x02(5);
    auto mr2= ublas::row (obsv2,0);
    x02.assign(mr2);


    ublas::vector<prec> init_para2(5);
    init_para2(0) = 0.004; /*alpha*/
    init_para2(1) = 0.5 ; /*beta*/
    init_para2(2) = 0.04; /*gamma*/
    init_para2(3) = 0.09; /*delta*/
    init_para2(4) = 0.08; /*mu*/

    ublas::identity_matrix<prec> B02 (5);

    kul::bfgs try2(obsv2, x02, T2, num_row - 1, init_para2, B02);
    result = try2.BFGS_heun();
    std::cout<<"result of observation2: "<<result<<std::endl;
    std::cout<<"LSE = "<< try2.lse_forward(result)<<std::endl;
    kul::write_prediction("../input/prediction2.out", result, x02);

    std::cout<<"!!!!my parameter order is alpha, beta, gamma, delta, mu!!!!"<<std::endl;

    return 0;


}