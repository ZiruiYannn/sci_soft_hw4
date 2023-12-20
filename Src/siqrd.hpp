#ifndef KUL_SIQRD_HPP
#define KUL_SIQRD_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cmath>

namespace ublas = boost::numeric::ublas;

namespace kul {
    template <typename S, typename P>
    auto siqrd_df (ublas::vector<S> const& x, P alpha, P beta, P gamma, P delta, P mu) {
        assert(x.size() == 5);

        typedef decltype( S() * P() ) value_type ;
        ublas::vector<value_type> df(5);

        df(0) = -beta * x(1) * x(0) / (x(0) + x(1) + x(3)) + mu * x(3);
        df(1) = (beta * x(0) / (x(0) +x(1) + x(3)) - gamma - delta - alpha) * x(1);
        df(2) = delta * x(1) - (gamma + alpha) * x(2);
        df(3) = gamma * (x(1) + x(2)) - mu * x(3);
        df(4) = alpha * (x(1) + x(2));

        return df;
    }

    template <typename S, typename P>
    auto siqrd_J (ublas::vector<S> const& x, P alpha, P beta, P gamma, P delta, P mu) {
        assert(x.size() == 5);

        typedef decltype( S() * P() ) value_type ;
        ublas::matrix<value_type> J(5,5);

        S SIR = x(0) + x(1) + x(3);
        S SIR2 = SIR * SIR;

        /*frac{\partial S}{\partial * }*/ 
        J(0,0) = - beta * (x(1)/SIR - x(0) * x(1) /SIR2 );
        J(0,1) = - beta * (x(0)/SIR - x(0) * x(1) /SIR2 );
        J(0,2) = 0;
        J(0,3) = - beta * ( - x(0) * x(1) /SIR2 ) + mu;
        J(0,4) = 0;
        /*frac{\partial I}{\partial *}*/ 
        J(1,0) = beta * (x(1)/SIR - x(0)*x(1)/SIR2);
        J(1,1) = beta * (x(0)/SIR - x(0) * x(1)/ SIR2) - gamma - delta - alpha;
        J(1,2) = 0;
        J(1,3) = beta * (- x(0)*x(1)/SIR2 );
        J(1,4) = 0;
        /*frac{\partial Q}{\partial *}*/
        J(2,0) = 0;
        J(2,1) = delta;
        J(2,2) = - gamma - alpha;
        J(2,3) = 0;
        J(2,4) = 0;
        /*frac{\partial R}{\partial *}*/
        J(3,0) = 0; 
        J(3,1) = gamma;
        J(3,2) = gamma;
        J(3,3) = - mu; 
        J(3,4) = 0; 
        /*frac{\partial D}{\partial *}*/
        J(4,0) = 0;
        J(4,1) = alpha; 
        J(4,2) = alpha; 
        J(4,3) = 0; 
        J(4,4) = 0;

        return J;
    }

}

#endif