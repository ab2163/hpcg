//CURRENTLY SAME AS REFERENCE IMPLEMENTATION
//BUT WITH DUMB REPLACEMENT OF REF FUNCTIONS WITH STDEXEC VERSIONS
//CHANGE THIS CODE!

#include "ComputeMG_stdexec.hpp"
#include "ComputeSYMGS_stdexec.hpp"
#include "ComputeSPMV_stdexec.hpp"
#include "ComputeRestriction_stdexec.hpp"
#include "ComputeProlongation_stdexec.hpp"
#include <cassert>
#include <iostream>

int ComputeMG_stdexec(const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_stdexec(A, r, x);
    if (ierr!=0) return ierr;
    ierr = ComputeSPMV_stdexec(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_stdexec(A, r);  if (ierr!=0) return ierr;
    ierr = ComputeMG_stdexec(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation_stdexec(A, x);  if (ierr!=0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS_stdexec(A, r, x);
    if (ierr!=0) return ierr;
  }
  else {
    ierr = ComputeSYMGS_stdexec(A, r, x);
    if (ierr!=0) return ierr;
  }
  return 0;
}

