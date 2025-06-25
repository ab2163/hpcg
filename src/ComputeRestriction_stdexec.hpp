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

auto ComputeRestriction_stdexec(double * time, const SparseMatrix & A, const Vector & rf){

  return stdexec::then([time, &](){ if(time != NULL) *time -= mytimer(); })
  | stdexec::bulk(stdexec::par, A.mgData->rc->localLength,
    [&](int i){ 
      double * Axfv = A.mgData->Axf->values;
      double * rfv = rf.values;
      double * rcv = A.mgData->rc->values;
      local_int_t * f2c = A.mgData->f2cOperator;
      rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
  })
  | stdexec::then([time, &](){ if(time != NULL) *time += mytimer(); });
}