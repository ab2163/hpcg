#ifndef ASSERT_INCLUDED
#define ASSERT_INCLUDED
#include <cassert>
#endif

#include "ComputeSYMGS_stdexec.hpp"
#include "ComputeSPMV_stdexec.hpp"
#include "ComputeRestriction_stdexec.hpp"
#include "ComputeProlongation_stdexec.hpp"

auto ComputeMG_stdexec(double * time, const SparseMatrix  & A, const Vector & r, Vector & x){
    
  if(A.mgData == 0){
    return stdexec::then([&, time](){
      if(time != NULL) *time -= mytimer();
      assert(x.localLength == A.localNumberOfColumns); //Make sure x contain space for halo values
      ZeroVector(x); //initialize x to zero
    })
    | ComputeSYMGS_stdexec(NULL, A, r, x)
    | stdexec::then([&, time](){
      if(time != NULL) *time += mytimer(); });
  }
  else return stdexec::then([&, time](){
      if(time != NULL) *time -= mytimer();
      assert(x.localLength == A.localNumberOfColumns); //Make sure x contain space for halo values
      ZeroVector(x); //initialize x to zero
    })
    //MUST FIND WAY OF HAVING VARIABLE NUMBER OF PRECONDITIONING STEPS!
    | ComputeSYMGS_stdexec(NULL, A, r, x)
    | ComputeSYMGS_stdexec(NULL, A, r, x)
    | ComputeSYMGS_stdexec(NULL, A, r, x)
    | ComputeSPMV_stdexec(NULL, A, x, *A.mgData->Axf)
    | ComputeRestriction_stdexec(NULL, A, r)
    | ComputeMG_stdexec(NULL, *A.Ac,*A.mgData->rc, *A.mgData->xc)
    | ComputeProlongation_stdexec(NULL, A, x)
    //MUST FIND WAY OF HAVING VARIABLE NUMBER OF POSTCONDITIONING STEPS!
    | ComputeSYMGS_stdexec(NULL, A, r, x)
    | ComputeSYMGS_stdexec(NULL, A, r, x)
    | ComputeSYMGS_stdexec(NULL, A, r, x)
    | stdexec::then([&, time](){
      if(time != NULL) *time += mytimer(); });
}

