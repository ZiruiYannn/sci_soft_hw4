#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cassert>
#include <iostream>
#include "bfgs.hpp"
#include "io_siqrd.hpp"
#include "siqrd.hpp"
#include <chrono>

namespace ublas = boost::numeric::ublas;
typedef double prec;

int main(int argc, char* argv[]) {
    int number_exp=200;
    int discard=5;

    double elapsed_time=0.;
    double average_time=0.;
    double squared_time=0.;
    double time_diff=0.;


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

    for(int exp=0;exp<number_exp+discard;exp++){
        auto t_start = std::chrono::high_resolution_clock::now();

        auto result = try1.BFGS_forward();

        auto t_end = std::chrono::high_resolution_clock::now();
        if(exp>=discard){
            elapsed_time=std::chrono::duration<double>(t_end-t_start).count(); 
            time_diff=elapsed_time-average_time;
            average_time+=time_diff/(exp-discard+1);
            squared_time+=time_diff*(elapsed_time-average_time);

            if (exp%20 == 0) {
                std::cout<<"result of observation1: "<<result<<std::endl;
            }
        }
    }

    std::cout<<"Time(s) for forward: "<<average_time<<" "<<std::sqrt(squared_time/(number_exp-1))<<std::endl;

    elapsed_time=0.;
    average_time=0.;
    squared_time=0.;
    time_diff=0.;

    for(int exp=0;exp<number_exp+discard;exp++){
        auto t_start = std::chrono::high_resolution_clock::now();

        auto result = try1.BFGS_heun();

        auto t_end = std::chrono::high_resolution_clock::now();
        if(exp>=discard){
            elapsed_time=std::chrono::duration<double>(t_end-t_start).count(); 
            time_diff=elapsed_time-average_time;
            average_time+=time_diff/(exp-discard+1);
            squared_time+=time_diff*(elapsed_time-average_time);

            if (exp%20 == 0) {
                std::cout<<"result of observation1: "<<result<<std::endl;
            }
        }
    }

    std::cout<<"Time(s) for heun: "<<average_time<<" "<<std::sqrt(squared_time/(number_exp-1))<<std::endl;

    elapsed_time=0.;
    average_time=0.;
    squared_time=0.;
    time_diff=0.;

    for(int exp=0;exp<number_exp+discard;exp++){
        auto t_start = std::chrono::high_resolution_clock::now();

        auto result = try1.BFGS_backward();

        auto t_end = std::chrono::high_resolution_clock::now();
        if(exp>=discard){
            elapsed_time=std::chrono::duration<double>(t_end-t_start).count(); 
            time_diff=elapsed_time-average_time;
            average_time+=time_diff/(exp-discard+1);
            squared_time+=time_diff*(elapsed_time-average_time);

            if (exp%20 == 0) {
                std::cout<<"result of observation1: "<<result<<std::endl;
            }
        }
    }

    std::cout<<"Time(s) for backward: "<<average_time<<" "<<std::sqrt(squared_time/(number_exp-1))<<std::endl;
  
    return 0;


}