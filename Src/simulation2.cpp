#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "ivp_solver.hpp"
#include "simple_ode.hpp"
#include "io_siqrd.hpp"
#include <type_traits>

namespace ublas = boost::numeric::ublas;
typedef double prec;

int main(int argc, char* argv[]) {
    
    assert(argc == 3);

    int N = atoi(argv[1]);
    assert(N > 0);

    prec T = atof(argv[2]);
    assert(T > 0);
    /*
    int N = 50000;
    prec T = 500;
    */

    ublas::vector<prec> init_x(50);
    std::generate(init_x.begin(), init_x.end(), [](){static int i=0; i++; return 0.01*i;});

    ublas::vector<prec> ms(50);
    std::generate(ms.begin(), ms.end(), [](){static int i=-1; i++; return 0.1*i;});

    /*
    std::cout<<init_x<<std::endl;
    std::cout<<ms<<std::endl;
    */

    auto df = [&ms] (auto const& x) {return kul::odes_df(x, ms);};
    auto J = [&ms] (auto const& x) {return kul::odes_J(x, ms);};

    
    kul::ivp ivp1(init_x, T, N, df, J);
    ublas::matrix<prec> out_mat = ivp1.simu_forward();
    kul::write_result("../output/fwe_simulation2.out", out_mat);

    out_mat = ivp1.simu_backward();
    kul::write_result("../output/bwe_simulation2.out", out_mat);

    out_mat = ivp1.simu_heun();
    kul::write_result("../output/heun_simulation2.out", out_mat);

    return 0;
}