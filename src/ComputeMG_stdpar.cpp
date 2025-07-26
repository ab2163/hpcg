#include "ComputeMG_stdpar.hpp"
#include "ComputeSYMGS_stdpar.hpp"
#include "ComputeSPMV_stdpar.hpp"
#include "ComputeRestriction_stdpar.hpp"
#include "ComputeProlongation_stdpar.hpp"
#include <cassert>

int ComputeMG_stdpar(const SparseMatrix &A, const Vector &r, Vector &x){
  assert(x.localLength == A.localNumberOfColumns); //make sure x contain space for halo values

  ZeroVector(x); //initialize x to zero

  int ierr = 0;
  if (A.mgData != 0){ //go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for(int i = 0; i < numberOfPresmootherSteps; i++) ierr += ComputeSYMGS_stdpar(A, r, x);
    if (ierr != 0) return ierr;
    ierr = ComputeSPMV_stdpar(A, x, *A.mgData->Axf); if (ierr != 0) return ierr;
    //perform restriction operation using simple injection
    ierr = ComputeRestriction_stdpar(A, r);  if (ierr != 0) return ierr;
    ierr = ComputeMG_stdpar(*A.Ac, *A.mgData->rc, *A.mgData->xc);  if (ierr != 0) return ierr;
    ierr = ComputeProlongation_stdpar(A, x);  if (ierr != 0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i = 0; i < numberOfPostsmootherSteps; i++) ierr += ComputeSYMGS_stdpar(A, r, x);
    if (ierr != 0) return ierr;
  }
  else{
    ierr = ComputeSYMGS_stdpar(A, r, x);
    if(ierr != 0) return ierr;
  }
  return 0;
}

