#include "../stdexec/include/stdexec/execution.hpp"
#include <__senders_core.hpp>
#include "ComputeProlongation_stdexec.hpp"

decltype(auto) ComputeProlongation_stdexec(stdexec::sender auto input, const SparseMatrix & Af, Vector & xf) {
  return stdexec::bulk(input, stdexec::par, Af.mgData->rc->localLength;,
    [&](int i){ 
      double * xfv = xf.values;
      double * xcv = Af.mgData->xc->values;
      local_int_t * f2c = Af.mgData->f2cOperator;
      xfv[f2c[i]] += xcv[i];
  });
}