#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "ivp_solver.hpp"
#include "siqrd.hpp"
#include "io_siqrd.hpp"

namespace ublas = boost::numeric::ublas;
typedef double prec;

int main(int argc, char* argv[]) {
    
    assert(argc == 3);

    int N = atoi(argv[1]);
    assert(N > 0);

    prec T = atof(argv[2]);
    assert(T > 0);
    /*
    int N = 100;
    prec T = 100;
    */

    prec beta, mu, gamma, alpha, delta;
    ublas::vector<prec> init_x(5);   /*initial S, I, Q, R, D*/

    kul::read_siqrd_para("../input/parameters.in", alpha, beta, gamma, delta, mu, init_x);    /*Read parameters from the file*/

    auto df = [&alpha, &beta, &gamma, &delta, &mu] (auto const& x) {return kul::siqrd_df(x, alpha, beta, gamma, delta, mu);};    /*pass function by lambda-expression*/
    auto J = [&alpha, &beta, &gamma, &delta, &mu] (auto const& x) {return kul::siqrd_J(x, alpha, beta, gamma, delta, mu);};    /*pass Jacobian by lambda-expression*/

    delta = 0;    /*case 1: using forward method*/
    kul::ivp ivp1(init_x, T, N, df, J);
    ublas::matrix<prec> out_mat = ivp1.simu_forward();
    kul::write_result("../output/fwe_no_measures.out", out_mat);
    
    delta = 0.2;    /*case 2: using backward method*/
    kul::ivp ivp2(init_x, T, N, df, J);
    out_mat = ivp2.simu_backward();
    kul::write_result("../output/bwe_quarantine.out", out_mat);

    delta = 0.9;    /*case 3: using heun method*/
    kul::ivp ivp3(init_x, T, N, df, J);
    out_mat = ivp3.simu_heun();
    kul::write_result("../output/heun_lockdown.out", out_mat);


    return 0;
}