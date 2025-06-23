#include <thread>
#include <cassert>
#include <iostream>

#include "ComputeWAXPBY_stdexec.hpp"
#include "mytimer.hpp"

template <stdexec::sender Sender>
auto ComputeWAXPBY_stdexec(Sender input, double & time, const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w) -> decltype(stdexec::then(input, [](){})){

  double t_begin = mytimer();
  assert(x.localLength>=n); //Test vector lengths
  assert(y.localLength>=n);
  const double * const xv = x.values;
  const double * const yv = y.values;
  double * const wv = w.values;

  if(alpha == 1.0) {
    return input | stdexec::bulk(start_point, stdexec::par, n, [&](local_int_t i){ wv[i] = xv[i] + beta*yv[i]; })
      | then([&](){ time += mytimer() - t_begin; });
  }
  else if(beta == 1.0){
    return input | stdexec::bulk(start_point, stdexec::par, n, [&](local_int_t i){ wv[i] = alpha*xv[i] + yv[i]; })
      | then([&](){ time += mytimer() - t_begin; });
  }
  else{
    return input | stdexec::bulk(start_point, stdexec::par, n, [&](local_int_t i){ wv[i] = alpha*xv[i] + beta*yv[i]; })
      | then([&](){ time += mytimer() - t_begin; });
  }
}
