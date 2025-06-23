#include "../stdexec/include/stdexec/execution.hpp"
#include <__senders_core.hpp>
#include "ComputeRestriction_stdexec.hpp"

decltype(auto) ComputeRestriction_stdexec(stdexec::sender auto input, const SparseMatrix & A, const Vector & rf){
  return stdexec::bulk(input, stdexec::par, A.mgData->rc->localLength,
    [&](int i){ 
      double * Axfv = A.mgData->Axf->values;
      double * rfv = rf.values;
      double * rcv = A.mgData->rc->values;
      local_int_t * f2c = A.mgData->f2cOperator;
      rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
  });
}
