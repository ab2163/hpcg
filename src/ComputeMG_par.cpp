#include "ComputeMG_par.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS_par.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include <cassert>

int ComputeMG_par(const SparseMatrix &A, const Vector &r, Vector &x){
  assert(x.localLength == A.localNumberOfColumns);

  ZeroVector(x); //initialize x to zero

  int ierr = 0;
  if(A.mgData != 0){ //go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for(int i = 0; i < numberOfPresmootherSteps; i++) ierr += ComputeSYMGS_par(A, r, x);
    if (ierr != 0) return ierr;
    ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); if (ierr != 0) return ierr;
    //perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);  if (ierr != 0) return ierr;
    ierr = ComputeMG_par(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr != 0) return ierr;
    ierr = ComputeProlongation_ref(A, x);  if (ierr != 0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for(int i = 0; i < numberOfPostsmootherSteps; i++) ierr += ComputeSYMGS_par(A, r, x);
    if (ierr != 0) return ierr;
  }
  else{
    ierr = ComputeSYMGS_par(A, r, x);
    if(ierr != 0) return ierr;
  }
  return 0;
}

