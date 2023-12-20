#ifndef KUL_BFGS_HPP
#define KUL_BFGS_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "siqrd.hpp"
#include "ivp_solver.hpp"
#include <cassert>

namespace ublas = boost::numeric::ublas;

namespace kul {

    template<typename P, typename Q, typename I, typename V, typename M>
    class bfgs {
        typedef decltype( Q() * P() ) value_type;
        typedef typename ublas::matrix<P>::size_type size_type;

        public:
        /*constructor and destructor*/
            bfgs(const ublas::matrix<P> & obsv, const ublas::vector<P> & x0, Q const t, I const n, V &init_para, M &B0) : \
            obsv_(obsv), t_(t), n_(n), dt_(t/n), x0_(x0), SIQRD_((std::pow(x0(0) + x0(1) + x0(2) + x0(3) + x0(4),2.4)) ), \
            init_para_(init_para), B0_(B0) {
                assert(obsv.size2() == 5);
                assert(obsv.size1() == n+1 );
                assert(x0.size() == 5);
                assert(init_para.size() == 5);
                assert(B0.size1() == 5);
                assert(B0.size2() == 5);
                eps_ = 1e-10;
                c1_ = 1e-4;
                tol_ = 1e-7;
                record_count_ = 8;
                init_step_size_ = 1;
                maxit_ = 1000;
            }

            ~bfgs() {}

        public:
        /*Get initial vector x0_*/
            ublas::vector<value_type> get_x0() {
                return x0_;
            }

        /*Get eps*/
            auto get_eps() {
                return eps_;
            }

        /*change the value of eps*/
            template <typename PP>
            void set_eps(PP eps) {
                eps_ = eps;
            }

        /*Get c1*/
            auto get_c1() {
                return c1_;
            }

        /*change the value of c1*/
            template <typename PP>
            void set_c1(PP c1) {
                c1_ = c1;
            }

        /*Get tol*/
            auto get_tol() {
                return tol_;
            }

        /*change the value of tol*/
            template <typename PP>
            void set_tol(PP tol) {
                tol_ = tol;
            }

        /*Get record_count*/
            auto get_count() {
                return record_count_;
            }

        /*change the value of record_count*/
            template <typename PP>
            void set_count(PP record_count) {
                record_count_ = record_count;
            }

        /*Get init_step_size*/
            auto get_init_step_size() {
                return tol_;
            }

        /*change the value of init_step_size*/
            template <typename PP>
            void set_init_step_size(PP init_step_size) {
                init_step_size_ = init_step_size;
            }

        /*Get maxit*/
            auto get_maxit() {
                return maxit_;
            }

        /*change the value of maxit*/
            template <typename PP>
            void set_maxit(PP maxit) {
                maxit_ = maxit;
            }

        /*Get SIQRD*/
            auto get_SIQRD() {
                return SIQRD_;
            }

        /*change the value of SIQRD*/
            template <typename PP>
            void set_SIQRD(PP siqrd) {
                SIQRD_ = siqrd;
            }

        /*get initial parameters*/
            auto get_init_para() {
                return init_para_;
            }

        private:
        /*Use prediction to compute LSE*/
            value_type lse(ublas::matrix<value_type> & pred) {
                assert(pred.size2() == 5);
                assert(pred.size1() == n_ + 1);
                value_type result = 0, temp;



                for (size_type i=1; i< n_ + 1; ++i) {
                    auto rm_obsv = ublas::row (obsv_, i);
                    auto rm_pred = ublas::row (pred, i);

                    auto new_vec = rm_pred - rm_obsv ;

                    temp = ublas::norm_2_square(new_vec);
                    temp /= t_ *SIQRD_;
                    result = result + temp;                
                }

                
                return result;
            }

        public:
        /*Compute LSE by 3 method. */
            value_type lse_forward(ublas::vector<value_type> const& p)  {
                
                auto df = [p] (auto const& x) {return kul::siqrd_df(x, p(0), p(1), p(2), p(3), p(4));};

                kul::ivp ivp1(x0_, t_, record_count_ * n_, df);
                ublas::matrix<value_type> pred = ivp1.simu_forward(false, record_count_);
                assert(pred.size1() == obsv_.size1());
                assert(pred.size2() == obsv_.size2());
                
                value_type result = lse(pred);

                return result;
            } 

            value_type lse_heun(ublas::vector<value_type> const& p)  {

                auto df = [p] (auto const& x) {return kul::siqrd_df(x, p(0), p(1), p(2), p(3), p(4));};
                
                kul::ivp ivp1(x0_, t_, record_count_ * n_, df);
                ublas::matrix<value_type> pred = ivp1.simu_heun(false, record_count_);
                assert(pred.size1() == obsv_.size1());
                assert(pred.size2() == obsv_.size2());
                value_type result = lse(pred);

                return result;
            } 

            value_type lse_backward(ublas::vector<value_type> const& p)  {
                
                auto df = [p] (auto const& x) {return kul::siqrd_df(x, p(0), p(1), p(2), p(3), p(4));};
                auto J = [p] (auto const& x) {return kul::siqrd_J(x, p(0), p(1), p(2), p(3), p(4));};

                kul::ivp ivp1(x0_, t_, record_count_ * n_, df, J);
                ublas::matrix<value_type> pred = ivp1.simu_backward(false, record_count_);
                assert(pred.size1() == obsv_.size1());
                assert(pred.size2() == obsv_.size2());
                value_type result = lse(pred);
                
                return result;
            } 

        private:
        /*gradient estimation*/
            ublas::vector<value_type> ge_lse_forward(ublas::vector<value_type> const& p) {
                V pp = p;
                ublas::vector<value_type> result(5);

                auto lse_old = lse_forward(p);

                for (I i = 0; i<5; i++) {
                    pp(i) = pp(i) *( 1 + eps_) ;
                    auto lse_new = lse_forward(pp) ;
                    result(i) = (lse_new - lse_old) / (pp(i) - p(i));

                    pp(i) = p(i);
                }

                return result;
            }

            ublas::vector<value_type> ge_lse_heun(ublas::vector<value_type> const& p) {
                V pp = p;
                ublas::vector<value_type> result(5);

                auto lse_old = lse_heun(p);

                for (I i = 0; i<5; i++) {
                    pp(i) = pp(i) *( 1 + eps_) ;
                    auto lse_new = lse_heun(pp) ;
                    result(i) = (lse_new - lse_old) / (pp(i) - p(i));
                    pp(i) = p(i);
                }

                return result;
            }

            ublas::vector<value_type> ge_lse_backward(ublas::vector<value_type> const& p) {
                V pp = p;
                ublas::vector<value_type> result(5);

                auto lse_old = lse_backward(p);

                for (I i = 0; i<5; i++) {
                    pp(i) = pp(i) *( 1 + eps_) ;
                    auto lse_new = lse_backward(pp) ;
                    result(i) = (lse_new - lse_old) / (pp(i) - p(i));
                    pp(i) = p(i);
                }

                return result;
            }

        private:
        /*line search*/
            value_type line_search_forward(ublas::vector<value_type> & p, ublas::vector<value_type> & di, ublas::vector<value_type> & ge_pi) {
                value_type eta = init_step_size_;
                I i = 0;

                value_type a, b, c, d;
                a = lse_forward(p + eta * di);
                c = lse_forward(p);
                d = c1_ * ublas::inner_prod(di, ge_pi);
                b = c +  eta* d;

                while (a > b) {
                    eta = eta/2;
                    a = lse_forward(p + eta * di);
                    b = c + eta* d;
                    ++i;
                    if (i>=maxit_) {
                        std::cout<<"line_search_forward reaches maximum iteration"<<std::endl;
                    }
                }

                return eta;
            }

            value_type line_search_heun(ublas::vector<value_type> & p, ublas::vector<value_type> & di, ublas::vector<value_type> & ge_pi) {
                value_type eta = init_step_size_;
                I i = 0;

                value_type a, b, c, d;
                a = lse_heun(p + eta * di);
                c = lse_heun(p);
                d = c1_ * ublas::inner_prod(di, ge_pi);
                b = c +  eta* d ;

                while (a > b) {
                    eta = eta/2;
                    a = lse_heun(p + eta * di);
                    b = c + eta* d;
                    ++i;
                    if (i>=maxit_) {
                        std::cout<<"line_search_heun reaches maximum iteration"<<std::endl;
                    }
                }

                return eta;
            }

            value_type line_search_backward(ublas::vector<value_type> & p, ublas::vector<value_type> & di, ublas::vector<value_type> & ge_pi) {
                value_type eta = init_step_size_;
                I i = 0;

                value_type a, b, c, d;
                a = lse_backward(p + eta * di);
                c = lse_backward(p);
                d = c1_ * ublas::inner_prod(di, ge_pi);
                b = c +  eta* d;

                while (a > b) {
                    eta = eta/2;
                    a = lse_backward(p + eta * di);
                    b = c + eta* d;
                    ++i;
                    if (i>=maxit_) {
                        std::cout<<"line_search_backward reaches maximum iteration"<<std::endl;
                    }
                }

                return eta;
            }

        public:
        /*Do BFGS algorithm to compute the optimal parameters*/
        /*B0 will be modified and turn into the estimated Hessian for the last round*/
            ublas::vector<value_type> BFGS_forward() {
                ublas::vector<value_type> di(5), s(5), pi1(5), Bs(5), y(5), ge_pi(5), ge_pi1(5);
                ublas::vector<value_type> pi = init_para_;
                value_type eta;
                ublas::matrix<value_type> B1 (5,5), B0 (5,5);
                /*I do not know why here i cannot use B1= B0_ (from identity matrix to matrix)*/
                for (I j = 0; j< 5; ++j) {
                    for (I i = 0; i< 5; ++i) {
                        B1(i,j) = B0_(i,j);
                    }
                }
                I i = 0;

                ge_pi1 = ge_lse_forward(pi);
                while (i< maxit_) {
                    B0 = B1;

                    di = ge_pi1;
                    ge_pi = ge_pi1;
                    kul::solve(B0, di);
                    di = -di;                  
                    eta = line_search_forward(pi, di, ge_pi);

                    if (eta*norm_2(di) / norm_2(pi) < tol_ ) {
                        break;
                    }

                    s = eta * di; 
                    pi1 = pi + s;
                    ge_pi1 = ge_lse_forward(pi1);

                    y =  ge_pi1 - ge_pi;

                    Bs = ublas::prod(B1, s);
                    B1 = B1 - ( ublas::outer_prod(Bs,Bs) / ublas::inner_prod(s,Bs) ) + ( ublas::outer_prod(y,y) / ublas::inner_prod(s,y) );

                    pi = pi1;
                    
                    ++i;
                }
                
                if (i >= maxit_) {
                    std::cout<<"BFGS_forward reaches maximum iteration. "<<std::endl;
                }

                return pi;
            }

            ublas::vector<value_type> BFGS_heun() {
                ublas::vector<value_type> di(5), s(5), pi1(5), Bs(5), y(5), ge_pi(5), ge_pi1(5);
                ublas::vector<value_type> pi = init_para_;
                value_type eta;
                ublas::matrix<value_type> B1 (5,5), B0 (5,5);
                /*I do not know why here i cannot use B1= B0_ (from identity matrix to matrix)*/
                for (I j = 0; j< 5; ++j) {
                    for (I i = 0; i< 5; ++i) {
                        B1(i,j) = B0_(i,j);
                    }
                }
                I i = 0;

                ge_pi1 = ge_lse_heun(pi);

                while (i< maxit_) {
                    B0 = B1;

                    di = ge_pi1;
                    ge_pi = ge_pi1;
                    kul::solve(B0, di);
                    di = -di;                     

                    eta = line_search_heun(pi, di, ge_pi);

                    if (eta*norm_2(di) / norm_2(pi) < tol_ ) {
                        break;
                    }
                    
                    s = eta * di; 
                    pi1 = pi + s;
                    ge_pi1 = ge_lse_heun(pi1);

                    y =  ge_pi1 - ge_pi;

                    Bs = ublas::prod(B1, s);
                    B1 = B1 - ( ublas::outer_prod(Bs,Bs) / ublas::inner_prod(s,Bs) ) + ( ublas::outer_prod(y,y) / ublas::inner_prod(s,y) );

                    pi = pi1;
                    
                    ++i;
                }
                
                if (i >= maxit_) {
                    std::cout<<"BFGS_heun reaches maximum iteration. "<<std::endl;
                }

                return pi;
            }

            ublas::vector<value_type> BFGS_backward() {
                ublas::vector<value_type> di(5), s(5), pi1(5), Bs(5), y(5), ge_pi(5), ge_pi1(5);
                ublas::vector<value_type> pi = init_para_;
                value_type eta;
                ublas::matrix<value_type> B1 (5,5), B0 (5,5);
                /*I do not know why here i cannot use B1= B0_ (from identity matrix to matrix)*/
                for (I j = 0; j< 5; ++j) {
                    for (I i = 0; i< 5; ++i) {
                        B1(i,j) = B0_(i,j);
                    }
                }
                I i = 0;

                ge_pi1 = ge_lse_backward(pi);

                while (i< maxit_) {
                    B0 = B1;

                    di = ge_pi1;
                    ge_pi = ge_pi1;
                    kul::solve(B0, di);
                    di = -di;                  
                    eta = line_search_backward(pi, di, ge_pi);

                    if (eta*norm_2(di) / norm_2(pi) < tol_ ) {
                        break;
                    }

                    s = eta * di; 
                    pi1 = pi + s;
                    ge_pi1 = ge_lse_backward(pi1);

                    y =  ge_pi1 - ge_pi;

                    Bs = ublas::prod(B1, s);
                    B1 = B1 - ( ublas::outer_prod(Bs,Bs) / ublas::inner_prod(s,Bs) ) + ( ublas::outer_prod(y,y) / ublas::inner_prod(s,y) );

                    pi = pi1;

                    
                    ++i;
                }
                
                if (i >= maxit_) {
                    std::cout<<"BFGS_backward reaches maximum iteration. "<<std::endl;
                }

                return pi;
            }

            private:
                ublas::matrix<value_type> const& obsv_;
                Q t_;
                I n_;
                Q dt_;
                ublas::vector<value_type> const& x0_; 
                value_type SIQRD_; /*SIQRD= S_i + I_i + Q_i + R_i + D_i should be constant, but due to numerical error, it might change. But as long
                as the result is meaningful, the error should be small.*/

                V const& init_para_;
                value_type eps_;
                value_type c1_;
                I record_count_;
                value_type tol_;
                value_type init_step_size_;
                M const& B0_;
                I maxit_;

    };

    
}



#endif