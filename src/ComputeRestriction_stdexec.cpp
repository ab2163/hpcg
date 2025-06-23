#include "ComputeRestriction_stdexec.hpp"
#include "mytimer.hpp"

template <stdexec::sender Sender>
auto ComputeRestriction_stdexec(Sender input, double & time, const SparseMatrix & A, const Vector & rf)
  -> decltype(stdexec::then(input, [](){})){

  double t_begin;

  return input | stdexec::then([&](){ t_begin = mytimer(); })
  | stdexec::bulk(input, stdexec::par, A.mgData->rc->localLength,
    [&](int i){ 
      double * Axfv = A.mgData->Axf->values;
      double * rfv = rf.values;
      double * rcv = A.mgData->rc->values;
      local_int_t * f2c = A.mgData->f2cOperator;
      rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
  });
  | stdexec::then([&](){ time += mytimer() - t_begin; });
}
