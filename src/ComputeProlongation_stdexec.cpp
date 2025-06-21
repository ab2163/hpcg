#include <cstdlib>

#include "../stdexec/include/stdexec/execution.hpp"
#include <__senders_core.hpp>
#include "ComputeProlongation_stdexec.hpp"

int ComputeProlongation_stdexec(stdexec::sender auto input, const SparseMatrix & Af, Vector & xf) {
  return input | then([](int input_success){
    //If the preceding sender did not execute properly then return a failure also
    if(input_success != 0){
      return EXIT_FAILURE;
    }
  })
  | stdexec::bulk(input, stdexec::par, Af.mgData->rc->localLength;,
    [&](int i){ 
      double * xfv = xf.values;
      double * xcv = Af.mgData->xc->values;
      local_int_t * f2c = Af.mgData->f2cOperator;
      xfv[f2c[i]] += xcv[i];
  }) 
  | then([](){ return 0; }); //return 0 for next sender in pipeline
}