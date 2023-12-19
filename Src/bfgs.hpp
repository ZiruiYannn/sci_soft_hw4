#ifndef KUL_BFGS_HPP
#define KUL_BFGS_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "siqrd.hpp"
#include <casssert>

namespace kul {

    template<typename P, typename Q, typename I>
    class bfgs {
        typedef P value_type;
        typedef ublas::matrix<P>::size_type size_type;

        public:
            bfgs(ublas::matrix<P> &const obsv, Q const t, I const n) : obsv_(obsv), t_(t), n_(n) {
                assert(obsv.size2() == 5);
                assert(obsv.size1() == n);
                assert(obsv(n,0) == t);
            }

            ~ivp() {}

        private:
            template <typename F, typename J>
            value_type lse(F df, J Jacob) const {
                
                value_type result=0;


                return result;
            } 

    }

    ublas::matrix<value_type> const& obsv_;
    Q t;
    I n;
}



#endif