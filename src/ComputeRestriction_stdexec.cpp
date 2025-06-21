#include <cstdlib>

#include "../stdexec/include/stdexec/execution.hpp"
#include <__senders_core.hpp>
#include "ComputeRestriction_stdexec.hpp"

int ComputeRestriction_stdexec(stdexec::sender auto input, const SparseMatrix & A, const Vector & rf){
  return input | then([](int input_success){
    //If the preceding sender did not execute properly then return a failure also
    if(input_success != 0){
      return EXIT_FAILURE;
    }
  })
  | stdexec::bulk(input, stdexec::par, A.mgData->rc->localLength,
    [&](int i){ 
      double * Axfv = A.mgData->Axf->values;
      double * rfv = rf.values;
      double * rcv = A.mgData->rc->values;
      local_int_t * f2c = A.mgData->f2cOperator;
      rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
  }) 
  | then([](){ return 0; }); //return 0 for next sender in pipeline
}
