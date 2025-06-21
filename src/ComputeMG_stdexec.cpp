#include <cassert>
#include <cstdlib>

#include "../stdexec/include/stdexec/execution.hpp"
#include <__senders_core.hpp>
#include "ComputeMG_stdexec.hpp"
#include "ComputeSYMGS_stdexec.hpp"
#include "ComputeSPMV_stdexec.hpp"
#include "ComputeRestriction_stdexec.hpp"
#include "ComputeProlongation_stdexec.hpp"

int ComputeMG_stdexec(stdexec::sender auto input, const SparseMatrix & A, const Vector & r, Vector & x){
  
  stdexec::sender auto current = input | then([&](){
    assert(x.localLength==A.localNumberOfColumns); //Make sure x contain space for halo values
    ZeroVector(x); //initialize x to zero
  });

  if (A.mgData != 0){
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i = 0; i < numberOfPresmootherSteps; ++i){
      //Add smoothing step to task graph multiple times
      current = ComputeSYMGS_stdexec(current, A, r, x);
    }

    current = ComputeSPMV_stdexec(current, A, x, *A.mgData->Axf);
    current = ComputeRestriction_stdexec(current, A, r);
    current = ComputeMG_stdexec(current, *A.Ac,*A.mgData->rc, *A.mgData->xc);
    current = ComputeProlongation_stdexec(current, A, x);

    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i){
      current = ComputeSYMGS_stdexec(current, A, r, x);
    }
  }
  else{
    current = ComputeSYMGS_stdexec(current, A, r, x);
  }

  return current;
}

