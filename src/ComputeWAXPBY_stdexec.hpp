#ifndef ASSERT_INCLUDED
#define ASSERT_INCLUDED
#include <cassert>
#endif

#ifndef EXECUTION_INCLUDED
#define EXECUTION_INCLUDED
#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#endif

#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED
#include "Vector.hpp"
#endif

#ifndef TIMER_INCLUDED
#define TIMER_INCLUDED
#include "mytimer.hpp"
#endif

auto ComputeWAXPBY_stdexec(double * time, const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w){

  assert(x.localLength >= n); //Test vector lengths
  assert(y.localLength >= n);
  const double * const xv = x.values;
  const double * const yv = y.values;
  double * const wv = w.values;

  if(alpha == 1.0) {
    return then([time, &](){ if(time != NULL) *time -= mytimer(); })
    | stdexec::bulk(stdexec::par, n, [beta, xv, wv, yv, &](local_int_t i){ 
        wv[i] = xv[i] + beta*yv[i]; })
    | then([time, &](){ if(time != NULL) *time += mytimer(); });
  }
  else if(beta == 1.0){
    return then([time, &](){ if(time != NULL) *time -= mytimer(); })
    | stdexec::bulk(stdexec::par, n, [alpha, xy, wv, yv, &](local_int_t i){ 
        wv[i] = alpha*xv[i] + yv[i]; })
    | then([time, &](){ if(time != NULL) *time += mytimer(); });
  }
  else{
    return then([time, &](){ if(time != NULL) *time -= mytimer(); })
    | stdexec::bulk(stdexec::par, n, [alpha, beta, xy, wv, yv, &](local_int_t i){ 
        wv[i] = alpha*xv[i] + beta*yv[i]; })
    | then([time, &](){ if(time != NULL) *time += mytimer(); });
  }
}