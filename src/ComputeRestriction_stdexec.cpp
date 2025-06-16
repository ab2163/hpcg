//CURRENTLY SAME AS REFERENCE IMPLEMENTATION
//CHANGE THIS CODE!

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction_stdexec.hpp"

int ComputeRestriction_stdexec(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i=0; i<nc; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

  return 0;
}
