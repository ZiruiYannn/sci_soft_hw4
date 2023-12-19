#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "ivp_solver.hpp"
#include "siqrd.hpp"
#include "io_siqrd.hpp"

namespace ublas = boost::numeric::ublas;

int main(int argc, char* argv[]) {
    
    assert(argc == 3);

    int N = atoi(argv[1]);
    assert(N > 0);

    double T = atof(argv[2]);
    assert(T > 0);
    /*
    int N = 100;
    double T = 100;
    */

    double beta, mu, gamma, alpha, delta;
    ublas::vector<double> init_x(5);

    kul::read_siqrd_para("../input/parameters.in", alpha, beta, gamma, delta, mu, init_x);

    auto df = [&alpha, &beta, &gamma, &delta, &mu] (auto const& x) {return kul::siqrd_df(x, alpha, beta, gamma, delta, mu);};
    auto J = [&alpha, &beta, &gamma, &delta, &mu] (auto const& x) {return kul::siqrd_J(x, alpha, beta, gamma, delta, mu);};

    delta = 0;
    kul::ivp ivp1(init_x, T, N, df, J);
    ublas::matrix<double> out_mat = ivp1.simu_forward();
    kul::write_result("../output/fwe_no_measures.out", out_mat);
    
    delta = 0.2;
    kul::ivp ivp2(init_x, T, N, df, J);
    out_mat = ivp2.simu_backward();
    kul::write_result("../output/bwe_quarantine.out", out_mat);

    delta = 0.9;
    kul::ivp ivp3(init_x, T, N, df, J);
    out_mat = ivp3.simu_heun();
    kul::write_result("../output/heun_lockdown.out", out_mat);






    return 0;
}