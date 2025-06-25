#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "mytimer.hpp"

auto ComputeProlongation_stdexec(double & time, const SparseMatrix & Af, Vector & xf){

  return stdexec::then([&](){ 
    if(time != NULL) time = mytimer(); })
  | stdexec::bulk(input, stdexec::par, Af.mgData->rc->localLength,
    [&](int i){
      double * xfv = xf.values;
      double * xcv = Af.mgData->xc->values;
      local_int_t * f2c = Af.mgData->f2cOperator;
      xfv[f2c[i]] += xcv[i];
  })
  | stdexec::then([&](){ 
    if(time != NULL) time = mytimer() - time; });
}