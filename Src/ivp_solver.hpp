#ifndef KUL_IVP_SOLVER_HPP
#define KUL_IVP_SOLVER_HPP

#include <type_traits>
#include <limits>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace ublas = boost::numeric::ublas

namespace kul {

    template<typename P>
    inline void solve (blas::matrix<P> & A, blas::vector<P> & b) {
        auto size_b = b.size();
        assert(A.size1() == size_b);
        assert(this->.size() == size_b);

        ublas::permutation_matrix<size_b> pm(size_b);
        ublas::lu_factorize(A,pm);
        ublas::lu_substitute(A, pm, b);
    }
    

    template<typname P, typename T, typename N, typename F, typename J, typename I>
    class ivp{
        public:
            typedef decltype( T() * P() ) value_type ;
            typedef ublas::vector<P>::size_type size_type;
            
        /*constructor/destructor*/
        public:
            ivp(ublas::vector<P> const& x0_, T const t_, N n_, F const& fun_, J const& Jac_ = 0, value_type tol_ = 1e-5, I maxit_ = 1000) : \
            x0(x0_), t(t_), n(n_), dt(t_/n_), fun(fun_), Jac(Jac_), tol(tol_), maxit(maxit_) {
                assert(tol > 0.);
                assert(tol*2 < std::numeric_limits<value_type>::max() );
                assert(max_it > 0);
            }

            ~ivp() {}

        public: 
            size_type x_size(){
                return x0.size();
            }

        /*Forward/Heun/Backward for one step*/
        /*These functions have some flexibility for users to use another time step or another IVP*/
        public:
            template<typename PP>
            ublas::vector<value_type> forward(ublas::vector<PP> const& x_, T dt_ = dt, F const& fun_ = fun) {
                assert(this->size() == b.size());

                ublas::vector<value_type> x = x_ + dt * fun_(x_);
                return x;
            }

            template<typename PP>
            ublas::vector<value_type> heun(ublas::vector<PP> const& x_, T dt_ = dt, F const& fun_ = fun) {
                assert(this->size() == b.size());

                ublas::vector<value_type> x = x_ + dt * (fun_(x_) / 2 + fun_(x_ + dt * fun_(x_)) / 2 );
                return x;
            }

            template<typename PP>
            ublas::vector<value_type> backward(ublas::vector<PP> const& x_, T dt_ = dt, F const& fun_ = fun, J const& Jac_ = Jac, value_type tol_ = 1e-5, I maxit_ = 1000) {
                assert(tol > 10*std::numeric_limits<value_type>::epsilon())
                assert(this->size() == b.size());
                assert(tol*2 < std::numeric_limits<value_type>::max());
                
                ublas::vector<value_type> x_new = x_, x_temp = x_, b(x_.size());
                ublas::identity_matrix<value_type> iden_mat(x_.size());
                ublas::matrix<value_type> A(x_.size(), x_.size());
                A = Jac_(x_new) * dt;
                A = A - iden_mat;
                b = x_ + dt * fun_(x_new) - x_new;

                value_type err = ublas::norm_2(b);
                I it_count = 0;

                while (err > tol * ublas::norm_2(x_) && it_count < maxit) {
                    x_temp = x_new;
                    solve(A,b);
                    x_new = x_new - b;

                    A = Jac_(x_new) * dt;
                    A = A - iden_mat;
                    b = x_ + dt * fun_(x_new) - x_new;

                    err = ublas::norm_2(b);

                    ++it_count;
                }

                if (it_count >= itmax) {
                    std::cout << "Reach maximum iteration " << maxit << std::endl;
                    std::cout << "Error = " << err << std::endl;
                    std::cout << "Tolerance = tol*norm2(x) =" << tol * ublas::norm_2(x_) << std::endl;
                }

                return x_new;
            }

        /*Here are the functions for solving IVP for three different method*/
        public:
            ublas::matrix<value_type>& simu_forward(){
                ublas::matrix<value_type> out_mat(n + 1 , 1 + x_size);
                out_mat(0,0) = 0;
                auto mr = ublas::row (out_mat, 0)
                auto mr1 = ublas::subrange(mr,1,x_size);
                mr1.assign(x0);

                auto x_temp = x0;

                for (size_type i = 1; i < n+1; i++ ) {
                    auto x_now = forward(x_temp);

                    out_mat(0,0) = 0;
                    auto mr = ublas::row (out_mat, 0)
                    auto mr1 = ublas::subrange(mr,1,x_size);
                    mr1.assign(x_now);

                    x_temp = x_now;
                }

                return out_mat;
            }

            ublas::matrix<value_type>& simu_heun(){
                ublas::matrix<value_type> out_mat(n + 1 , 1 + x_size);
                out_mat(0,0) = 0;
                auto mr = ublas::row (out_mat, 0)
                auto mr1 = ublas::subrange(mr,1,x_size);
                mr1.assign(x0);

                auto x_temp = x0;

                for (size_type i = 1; i < n+1; i++ ) {
                    auto x_now = heun(x_temp);

                    out_mat(0,0) = 0;
                    auto mr = ublas::row (out_mat, 0)
                    auto mr1 = ublas::subrange(mr,1,x_size);
                    mr1.assign(x_now);

                    x_temp = x_now;
                }

                return out_mat;
            }

            ublas::matrix<value_type>& simu_backward(){
                ublas::matrix<value_type> out_mat(n + 1 , 1 + x_size);
                out_mat(0,0) = 0;
                auto mr = ublas::row (out_mat, 0)
                auto mr1 = ublas::subrange(mr,1,x_size);
                mr1.assign(x0);

                auto x_temp = x0;

                for (size_type i = 1; i < n+1; i++ ) {
                    auto x_now = backward(x_temp);

                    out_mat(0,0) = 0;
                    auto mr = ublas::row (out_mat, 0)
                    auto mr1 = ublas::subrange(mr,1,x_size);
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


    }

}

#endif