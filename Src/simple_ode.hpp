#ifndef KUL_DECP_ODE_HPP
#define KUL_DECP_ODE_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cmath>
#include <type_traits>
#include <algorithm>

namespace ublas = boost::numeric::ublas;

namespace kul {
    
    template <typename T, typename P>
    auto simple_df (T x, P m) {
        return -10 * std::pow(x - m,3.0);
    }

    template <typename T, typename P>
    auto simple_J (T x, P m) {
        return -30 * std::pow(x-m, 2.0);
    }

    template <typename S, typename P>
    auto  odes_df (ublas::vector<S> const& x, ublas::vector<P> const& ms) {
        assert(x.size() == ms.size());

        typedef decltype( S() * P() ) value_type ;
        ublas::vector<value_type> df(x.size());

        std::transform(x.begin(), x.end(), ms.begin(), df.begin(), [](S c, P d) {return simple_df(c,d);});
              
        return df;
    }

    template <typename S, typename P>
    auto odes_J (ublas::vector<S> const& x, ublas::vector<P> const& ms) {
        assert(x.size() == ms.size());

        typedef decltype( S() * P() ) value_type ;
        ublas::vector<value_type> J(x.size());

        std::transform(x.begin(), x.end(), ms.begin(), J.begin(), [](S c, P d) {return simple_J(c,d);});

        return J;
    }


}

#endif