#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

typedef double prec;

namespace ublas = boost::numeric::ublas;
int main(int argc, char *argv[]){
    ublas::matrix<prec> A(500,500,27.);
    ublas::vector<prec> v(200,1.);
    auto mr= ublas::row (A,0);
    auto mr1 = ublas::subrange(mr,1,5);
    std::cout<<mr1<<std::endl;
    for (unsigned int i = 0; i < 500; i++)
    {
    auto mr= ublas::row (A,i);
    auto mr1 = ublas::subrange(mr,1,201);
    mr1.assign(v*(i+1));
    }
    std::cout<<A(0,3)<<" "<<A(1,3)<<" "<<A(200,3)<<std::endl;
    return 0;
}