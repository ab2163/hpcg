#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "mytimer.hpp"

auto ComputeRestriction_stdexec(double & time, const SparseMatrix & A, const Vector & rf){

  return stdexec::then([&](){
    if(time != NULL) time = mytimer(); })
  | stdexec::bulk(input, stdexec::par, A.mgData->rc->localLength,
    [&](int i){ 
      double * Axfv = A.mgData->Axf->values;
      double * rfv = rf.values;
      double * rcv = A.mgData->rc->values;
      local_int_t * f2c = A.mgData->f2cOperator;
      rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
  })
  | stdexec::then([&](){ 
    if(time != NULL) time = mytimer() - time; });
}