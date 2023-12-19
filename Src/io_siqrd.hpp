#ifndef KUL_IO_SIQRD_HPP
#define KUL_IO_SIQRD_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace kul {
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

}


#endif