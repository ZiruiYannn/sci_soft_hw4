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

    /*solve Ax=b. */
    /*Output: b is the result x=A^-1b*/
    template<typename P, typename Q>
    inline void solve (ublas::matrix<P> & A, ublas::vector<Q> & b) {
        assert(A.size1() == b.size());

        ublas::permutation_matrix<size_t> pm(A.size1());
        ublas::lu_factorize(A,pm);
        ublas::lu_substitute(A, pm, b);
    }

    /*solve Ax=b. When A is identity matrix*/
    /*Output: b is the result x=A^-1b*/
    template<typename P, typename Q>
    inline void solve (ublas::identity_matrix<P> & A, ublas::vector<Q> & b) {
        assert(A.size1() == b.size());

        for (decltype(ublas::vector<Q>::size_type) i = 0; i < b.size(); ++i) {
            b(i) = b(i) / A(i,i);
        }
    }

    /*concept for diagonised operator*/
    template <typename Op>
    concept OpDiag = requires (Op & op, ublas::vector<double>& p, ublas::vector<double> const& q) {
        p = op(q);
    };

    /*concept for normal operator*/
    template <typename Op>
    concept OpMat = requires (Op & op, ublas::matrix<double>& p, ublas::vector<double> const& q) {
        p = op(q);
        (op(q)).size1() == (op(q)).size2();
    };
    
    /*class ivp*/
    template<typename P, typename T, typename N, typename F, typename J=int, typename I = int>
    class ivp{
        public:
            typedef decltype( T() * P() ) value_type ;
            
        /*constructor/destructor*/
        public:
            ivp(ublas::vector<P> const& x0_, T const t_, N n_, F const& fun_, J const& Jac_ = 0, value_type tol_ = 1e-5, I maxit_ = 1000) : \
            x0(x0_), t(t_), n(n_), fun(fun_), Jac(Jac_), tol(tol_), maxit(maxit_) {
                assert(tol_ > 0.);
                assert(tol_*2 < std::numeric_limits<value_type>::max() );
                assert(maxit_ > 0);
                dt = t_/n_;
            }

            ~ivp() {}

        public: 
        /*get the size of the vector of initial values */
            auto x_size(){
                return x0.size();
            }

        /*get dt*/
            auto get_dt(){
                return dt;
            }
        
        private:
        /*Forward/Heun/Backward for one step*/
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

            /*backward for normal Jacobian*/
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

            /*backward for diagonised Jacobian*/
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

        public:
        /*Here are the functions for users to solve IVP for three different method*/
        /*simu_forward([flag]): use forward method. */
        /*simu_heun([flag]): use heun method. */
        /*simu_backward([flag]): use backward method. */
        /*If flag = true (default), then the first column of output matrix is the time; 
        else it does not contains the column of time and uses record_count (mainly for the task2 estimation).*/
            ublas::matrix<value_type> simu_forward(bool flag=true, I const record_count = 1) {
                decltype(x_size()) num_col = x_size(), num_row;
                if (flag) {
                    num_col = num_col + 1;
                    num_row = n + 1;
                } else {
                    num_row = floor(n/record_count) + 1;
                }
                ublas::matrix<value_type> out_mat( num_row , num_col);

                if (flag) {
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
                } else {
                    auto mr = ublas::row (out_mat, 0);
                    mr.assign(x0);

                    auto x_temp = x0;
                    I j = 0;

                    for (decltype(x0.size()) i = 1; i < n+1; i++ ) {
                        ++j;

                        auto x_now = forward(x_temp);
                        if (j >= record_count) {
                            auto mr = ublas::row (out_mat, floor(i/record_count));
                            mr.assign(x_now);
                            j = 0;
                        }

                        x_temp = x_now;
                    }
                }
                    
                return out_mat;
            }

            ublas::matrix<value_type> simu_heun(bool flag=true, I const record_count = 1){
                decltype(x_size()) num_col = x_size(), num_row;
                if (flag) {
                    num_col = num_col + 1;
                    num_row = n + 1;
                } else {
                    num_row = floor(n/record_count) + 1;
                }
                ublas::matrix<value_type> out_mat( num_row , num_col);

                if (flag) {
                    out_mat(0,0) = 0;
                    auto mr = ublas::row (out_mat, 0);
                    auto mr1 = ublas::subrange(mr,1, 1 + x_size());
                    mr1.assign(x0);

                    auto x_temp = x0;

                    for (decltype(x0.size()) i = 1; i < n+1; i++ ) {
                        auto x_now = heun(x_temp);

                        out_mat(i,0) = i * dt;
                        auto mr = ublas::row (out_mat, i);
                        auto mr1 = ublas::subrange(mr,1, 1 + x_size());
                        mr1.assign(x_now);

                        x_temp = x_now;
                    }
                } else {
                    auto mr = ublas::row (out_mat, 0);
                    mr.assign(x0);

                    auto x_temp = x0;
                    I j = 0;

                    for (decltype(x0.size()) i = 1; i < n+1; i++ ) {
                        ++j;

                        auto x_now = heun(x_temp);

                        if (j >= record_count) {
                            auto mr = ublas::row (out_mat, floor(i/record_count));
                            mr.assign(x_now);
                            j = 0;
                        }

                        x_temp = x_now;
                    }
                }
                    
                return out_mat;
            }

            ublas::matrix<value_type> simu_backward(bool flag=true, I const record_count = 1){
                decltype(x_size()) num_col = x_size(), num_row;
                if (flag) {
                    num_col = num_col + 1;
                    num_row = n + 1;
                } else {
                    num_row = floor(n/record_count) + 1;
                }
                ublas::matrix<value_type> out_mat( num_row , num_col);

                if (flag) {
                    out_mat(0,0) = 0;
                    auto mr = ublas::row (out_mat, 0);
                    auto mr1 = ublas::subrange(mr,1, 1 + x_size());
                    mr1.assign(x0);

                    auto x_temp = x0;

                    for (decltype(x0.size()) i = 1; i < n+1; i++ ) {
                        auto x_now = backward(x_temp);

                        out_mat(i,0) = i * dt;
                        auto mr = ublas::row (out_mat, i);
                        auto mr1 = ublas::subrange(mr,1, 1 + x_size());
                        mr1.assign(x_now);

                        x_temp = x_now;
                    }
                } else {
                    auto mr = ublas::row (out_mat, 0);
                    mr.assign(x0);

                    auto x_temp = x0;
                    I j = 0;

                    for (decltype(x0.size()) i = 1; i < n+1; i++ ) {
                        ++j;
                        auto x_now = backward(x_temp);

                        if (j >= record_count) {
                            auto mr = ublas::row (out_mat, floor(i/record_count));
                            mr.assign(x_now);
                            j = 0;
                        }

                        x_temp = x_now;
                    }
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