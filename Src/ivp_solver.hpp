#ifndef KUL_IVP_SOLVER_HPP
#define KUL_IVP_SOLVER_HPP

#include <cassert>
#include <type_traits>
#include <limits>
#include <iostream>
#include <fstream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <algorithm>

namespace ublas = boost::numeric::ublas;

namespace kul {

    template<typename P, typename Q>
    inline void solve (ublas::matrix<P> & A, ublas::vector<Q> & b) {
        assert(A.size1() == b.size());

        ublas::permutation_matrix<size_t> pm(A.size1());
        ublas::lu_factorize(A,pm);
        ublas::lu_substitute(A, pm, b);
    }


    template <typename Op>
    concept OpDiag = requires (Op & op, ublas::vector<double>& p, ublas::vector<double> const& q) {
        p = op(q);
    };

    template <typename Op>
    concept OpMat = requires (Op & op, ublas::matrix<double>& p, ublas::vector<double> const& q) {
        p = op(q);
        (op(q)).size1() == (op(q)).size2();
    };
    

    template<typename P, typename T, typename N, typename F, typename J, typename I = int>
    class ivp{
        public:
            typedef decltype( T() * P() ) value_type ;
            
        /*constructor/destructor*/
        public:
            ivp(ublas::vector<P> const& x0_, T const t_, N n_, F const& fun_, J const& Jac_ = [](ublas::vector<P> p) {return 0;}, value_type tol_ = 1e-5, I maxit_ = 1000) : \
            x0(x0_), t(t_), n(n_), fun(fun_), Jac(Jac_), tol(tol_), maxit(maxit_) {
                assert(tol_ > 0.);
                assert(tol_*2 < std::numeric_limits<value_type>::max() );
                assert(maxit_ > 0);
                dt = t_/n_;
            }

            ~ivp() {}

        public: 
            auto x_size(){
                return x0.size();
            }

        /*Forward/Heun/Backward for one step*/
        private:
            template<typename PP>
            ublas::vector<value_type> forward(ublas::vector<PP> const& x_) {
                assert(x_size() == x_.size());

                ublas::vector<value_type> x = x_ + dt * fun(x_);
                return x;
            }

            template<typename PP>
            ublas::vector<value_type> heun(ublas::vector<PP> const& x_) {
                assert(x_size() == x_.size());

                ublas::vector<value_type> x = x_ + dt * fun(x_);
                x = x_ + dt * (fun(x_) / 2 + fun(x) / 2 );
                return x;
            }

            template<typename PP> requires OpMat<J>
            inline decltype(auto) backward(ublas::vector<PP> const& x_) {
                assert(x_size() == x_.size());
                
                ublas::vector<value_type> x_new = x_, x_temp = x_, b(x_.size());
                ublas::identity_matrix<value_type> iden_mat(x_.size());
                ublas::matrix<value_type> A(x_.size(), x_.size());
                A = Jac(x_) * dt;
                A = A - iden_mat;
                b = x_ + dt * fun(x_new) - x_new;

                value_type err = ublas::norm_2(b);
                I it_count = 0;

                while (err > tol * ublas::norm_2(x_) && it_count < maxit) {
                    x_temp = x_new;
                    solve(A,b);
                    x_new = x_new - b;

                    A = Jac(x_new) * dt;
                    A = A - iden_mat;
                    b = x_ + dt * fun(x_new) - x_new;

                    err = ublas::norm_2(b);

                    ++it_count;
                }

                if (it_count >= maxit) {
                    std::cout << "Reach maximum iteration " << maxit << std::endl;
                    std::cout << "Error = " << err << std::endl;
                    std::cout << "Tolerance = tol*norm2(x) =" << tol * ublas::norm_2(x_) << std::endl;
                }

                return x_new;
            }

            template<typename PP> requires OpDiag<J>
            inline decltype(auto) backward(ublas::vector<PP> const& x_) {
                assert(x_size() == x_.size());
                
                ublas::vector<value_type> x_new = x_, x_temp = x_, b(x_.size());
                ublas::vector<value_type> iden_mat(x_.size(),1.0);
                ublas::vector<value_type> A(x_.size());
                A = Jac(x_) * dt;
                A = A - iden_mat;
                b = x_ + dt * fun(x_new) - x_new;

                value_type err = ublas::norm_2(b);
                I it_count = 0;

                while (err > tol * ublas::norm_2(x_) && it_count < maxit) {
                    x_temp = x_new;
                    std::transform(A.begin(), A.end(), b.begin(), b.begin(), [](value_type c, value_type d) {return d/c;});
                    x_new = x_new - b;

                    A = Jac(x_new) * dt;
                    A = A - iden_mat;
                    b = x_ + dt * fun(x_new) - x_new;

                    err = ublas::norm_2(b);

                    ++it_count;
                }

                if (it_count >= maxit) {
                    std::cout << "Reach maximum iteration " << maxit << std::endl;
                    std::cout << "Error = " << err << std::endl;
                    std::cout << "Tolerance = tol*norm2(x) =" << tol * ublas::norm_2(x_) << std::endl;
                }

                return x_new;
            }

        /*Here are the functions for solving IVP for three different method*/
        public:
            ublas::matrix<value_type> simu_forward() {
                ublas::matrix<value_type> out_mat(n + 1 , 1 + x_size());
                out_mat(0,0) = 0;
                auto mr = ublas::row (out_mat, 0);
                auto mr1 = ublas::subrange(mr,1, 1 + x_size());
                mr1.assign(x0);

                auto x_temp = x0;

                for (decltype(x0.size()) i = 1; i < n+1; i++ ) {
                    auto x_now = forward(x_temp);

                    out_mat(i,0) = i * dt;
                    auto mr = ublas::row (out_mat, i);
                    auto mr1 = ublas::subrange(mr,1, 1 + x_size());
                    mr1.assign(x_now);

                    x_temp = x_now;
                }


                return out_mat;
            }

            ublas::matrix<value_type> simu_heun(){
                ublas::matrix<value_type> out_mat(n + 1 , 1 + x_size());
                out_mat(0,0) = 0;
                auto mr = ublas::row (out_mat, 0);
                auto mr1 = ublas::subrange(mr,1, 1 + x_size());
                mr1.assign(x0);

                auto x_temp = x0;

                for (decltype(x0.size()) i = 1; i < n+1; i++ ) {
                    auto x_now = heun(x_temp);

                    out_mat(i,0) = i * dt;
                    auto mr = ublas::row (out_mat, i);
                    auto mr1 = ublas::subrange(mr,1,1 + x_size());
                    mr1.assign(x_now);

                    x_temp = x_now;
                }

                return out_mat;
            }

            ublas::matrix<value_type> simu_backward(){
                ublas::matri#endif
                auto x_temp = x0;


                for (decltype(x0.size()) i = 1; i < n+1; i++ ) {
                    auto x_now = backward(x_temp);

                    out_mat(i,0) = i * dt;
                    auto mr = ublas::row (out_mat, i);
                    auto mr1 = ublas::subrange(mr,1,1 + x_size());
                    mr1.assign(x_now);
                    x_temp = x_now;
                }

                return out_mat;
            }

        private:
            ublas::vector<value_type> const& x0;
            T t;
            N n;
            T dt;
            F const& fun;
            J const& Jac;
            value_type tol;
            I maxit;


    };

    
    

}

#endif