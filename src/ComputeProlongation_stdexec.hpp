#ifndef EXECUTION_INCLUDED
#define EXECUTION_INCLUDED
#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#endif

#ifndef SPARSE_INCLUDED
#define SPARSE_INCLUDED
#include "SparseMatrix.hpp"
#endif

#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED
#include "Vector.hpp"
#endif

#ifndef TIMER_INCLUDED
#define TIMER_INCLUDED
#include "mytimer.hpp"
#endif

auto ComputeProlongation_stdexec(double * time, const SparseMatrix & Af, Vector & xf){

  return stdexec::then([time, &](){ if(time != NULL) *time -= mytimer(); })
  | stdexec::bulk(stdexec::par, Af.mgData->rc->localLength,
    [&](int i){
      double * xfv = xf.values;
      double * xcv = Af.mgData->xc->values;
      local_int_t * f2c = Af.mgData->f2cOperator;
      xfv[f2c[i]] += xcv[i];
  })
  | stdexec::then([time, &](){ if(time != NULL) *time += mytimer(); });
}