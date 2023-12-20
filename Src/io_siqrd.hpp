#ifndef KUL_IO_SIQRD_HPP
#define KUL_IO_SIQRD_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace kul {
    /*read parameters from a file*/
    template<typename F, typename P, typename Q>
    void read_siqrd_para(F const filename, P &alpha, P &beta, P &gamma, P &delta, P &mu, ublas::vector<Q> &init_x) {
        std::ifstream para_file(filename);

        std::string line;
        std::getline(para_file, line);

        std::istringstream ss(line);

        Q S0, I0;
        ss >> beta >> mu >> gamma >> alpha >> delta >> S0 >> I0;

        para_file.close();

        init_x(0) = S0;
        init_x(1) = I0;
        init_x(2) = 0;
        init_x(3) = 0;
        init_x(4) = 0;
    }

    /*Before reading the observation matrix from a file, use observation_get_size to get the size of this matrix*/
    template<typename F, typename I>
    void observation_get_size(F const filename, I &num_row, I &num_col) {
        std::ifstream para_file(filename);

        std::string line;
        std::getline(para_file, line);

        std::istringstream ss(line);

        ss >> num_row >> num_col;

        para_file.close();
    }

    /*Read the observation matrix, simulation time. */
    template<typename F, typename P, typename T, typename I>
    void read_observation(F const filename, ublas::matrix<P> & obsv, T &t, I num_row, I num_col) {
        std::ifstream para_file(filename);

        std::string line;
        std::getline(para_file, line);

        for (decltype(obsv.size1()) i = 0; i < num_row; ++i) {
            para_file >>t;
            for (decltype(obsv.size1()) j = 0; j < num_col; ++j) {
                para_file >> obsv(i, j);
            }
        }

        para_file.close();
    }

    /*write the simulation result to a file.*/
    template<typename F, typename Q>
    void write_result(F const filename, ublas::matrix<Q> const& out_mat) {
        std::ofstream wfile(filename);
        for (decltype(out_mat.size1()) i = 0; i < out_mat.size1() ; i++) {
            for (decltype(out_mat.size2()) j = 0; j < out_mat.size2(); j++) {
                wfile << out_mat(i, j) << "\t"; // seperate with Tab
            }
            wfile << std::endl; // write on New line
        }
        wfile.close();
    }

    template <typename F, typename P, typename Q>
    void write_prediction(F const filename, ublas::vector<P> const& para_set,ublas::vector<Q> const& x0)  {
        std::ofstream wfile(filename);
        wfile << para_set(1)<< "\t"<<para_set(4)<< "\t"<<para_set(2)<< "\t"<<para_set(0)<< "\t"<<para_set(3)<< "\t"<<x0(0)<< "\t"<<x0(1);
        wfile.close();
    }

}


#endif