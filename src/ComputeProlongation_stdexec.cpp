#include "ComputeProlongation_stdexec.hpp"
#include "mytimer.hpp"

template <stdexec::sender Sender>
auto ComputeProlongation_stdexec(Sender input, double & time, const SparseMatrix & Af, Vector & xf)
  -> decltype(stdexec::then(input, [](){})){

  double t_begin;

  return input | stdexec::then([&](){ t_begin = mytimer(); })
  | stdexec::bulk(input, stdexec::par, Af.mgData->rc->localLength,
    [&](int i){
      double * xfv = xf.values;
      double * xcv = Af.mgData->xc->values;
      local_int_t * f2c = Af.mgData->f2cOperator;
      xfv[f2c[i]] += xcv[i];
  });
  | stdexec::then([&](){ time += mytimer() - t_begin; });
}