#include <cassert>
#include <algorithm>
#include <execution>

#include "ComputeWAXPBY_stdpar.hpp"

int ComputeWAXPBY_stdpar(const local_int_t n, const double alpha, const Vector &x,
    const double beta, const Vector &y, Vector &w){

  assert(x.localLength >= n); //test vector lengths
  assert(y.localLength >= n);

  const double * const xv = x.values;
  const double * const yv = y.values;
  double * const wv = w.values;

  if (alpha == 1.0){
    std::transform(std::execution::par_unseq, xv, xv + n, yv, wv, [&](double xi, double yi){ return xi + beta*yi; });
  }else if (beta == 1.0){
    std::transform(std::execution::par_unseq, xv, xv + n, yv, wv, [&](double xi, double yi){ return alpha*xi + yi; });
  }else {
    std::transform(std::execution::par_unseq, xv, xv + n, yv, wv, [&](double xi, double yi){ return alpha*xi + beta*yi; });
  }

  return 0;
}
