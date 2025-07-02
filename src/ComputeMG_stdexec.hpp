#ifndef ASSERT_INCLUDED
#define ASSERT_INCLUDED
#include <cassert>
#endif

#include "ComputeSYMGS_stdexec.hpp"
#include "ComputeSPMV_stdexec.hpp"
#include "ComputeRestriction_stdexec.hpp"
#include "ComputeProlongation_stdexec.hpp"

auto preRecursionMG(const SparseMatrix  & A, const Vector & r, Vector & x){
  //MUST FIND WAY OF HAVING VARIABLE NUMBER OF PRECONDITIONING STEPS!
  return ComputeSYMGS_stdexec(NULL, A, r, x)
  | ComputeSYMGS_stdexec(NULL, A, r, x)
  | ComputeSYMGS_stdexec(NULL, A, r, x)
  | ComputeSPMV_stdexec(NULL, A, x, *A.mgData->Axf)
  | ComputeRestriction_stdexec(NULL, A, r);
}

auto postRecursionMG(const SparseMatrix  & A, const Vector & r, Vector & x){
  return ComputeProlongation_stdexec(NULL, A, x)
  //MUST FIND WAY OF HAVING VARIABLE NUMBER OF POSTCONDITIONING STEPS!
  | ComputeSYMGS_stdexec(NULL, A, r, x)
  | ComputeSYMGS_stdexec(NULL, A, r, x)
  | ComputeSYMGS_stdexec(NULL, A, r, x);
}

auto terminalMG(const SparseMatrix  & A, const Vector & r, Vector & x){
  return ComputeSYMGS_stdexec(NULL, A, r, x);
}

auto ComputeMG_stdexec(double * time, const SparseMatrix  & A, const Vector & r, Vector & x){

  //IS THERE A RISK OF THESE BEING UPDATED BY A SUBSEQUENT CALL
  //BEFORE THE FIRST CALL FINISHES??
  static std::vector<const SparseMatrix*> matrix_ptrs(4);
  static std::vector<const Vector*> res_ptrs(4);
  static std::vector<Vector*> xval_ptrs(4);

  matrix_ptrs[0] = &A;
  res_ptrs[0] = &r;
  xval_ptrs[0] = &x;
  for(int cnt = 1; cnt < 4; cnt++){
    matrix_ptrs[cnt] = matrix_ptrs[cnt - 1]->Ac;
    res_ptrs[cnt] = matrix_ptrs[cnt - 1]->mgData->rc;
    xval_ptrs[cnt] = matrix_ptrs[cnt - 1]->mgData->xc;
  }

  return stdexec::then([&, time](){
    if(time != NULL) *time -= mytimer();
    
    assert(x.localLength == A.localNumberOfColumns); //Make sure x contain space for halo values
    ZeroVector(x); //initialize x to zero
    
    //ComputeMG_ref(A, r, x);
  })
  
  | preRecursionMG(*matrix_ptrs[0], *res_ptrs[0], *xval_ptrs[0])
  | preRecursionMG(*matrix_ptrs[1], *res_ptrs[1], *xval_ptrs[1])
  | preRecursionMG(*matrix_ptrs[2], *res_ptrs[2], *xval_ptrs[2])
  | terminalMG(*matrix_ptrs[3], *res_ptrs[3], *xval_ptrs[3])
  | postRecursionMG(*matrix_ptrs[2], *res_ptrs[2], *xval_ptrs[2])
  | postRecursionMG(*matrix_ptrs[1], *res_ptrs[1], *xval_ptrs[1])
  | postRecursionMG(*matrix_ptrs[0], *res_ptrs[0], *xval_ptrs[0])
  
  | stdexec::then([&, time](){
      if(time != NULL) *time += mytimer(); });
}

