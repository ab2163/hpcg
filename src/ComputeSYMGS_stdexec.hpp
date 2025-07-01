#ifndef ASSERT_INCLUDED
#define ASSERT_INCLUDED
#include <cassert>
#endif

#ifndef EXECUTION_INCLUDED
#define EXECUTION_INCLUDED
#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#endif

#ifndef SPARSE_INCLUDED
#define SPARSE_INCLUDED
#include "SparseMatrix.hpp"
#endif

#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED
#include "Vector.hpp"
#endif

#ifndef TIMER_INCLUDED
#define TIMER_INCLUDED
#include "mytimer.hpp"
#endif

#include "ComputeSYMGS_ref.hpp"

auto ComputeSYMGS_stdexec(double * time, const SparseMatrix & A, const Vector & r, Vector & x){

  return stdexec::then([&, time](){
    if(time != NULL) *time -= mytimer();
    ComputeSYMGS_ref(A, r, x);
    /*
    assert(x.localLength == A.localNumberOfColumns); //Make sure x contain space for halo values
#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif

    const local_int_t nrow = A.localNumberOfRows;
    double ** matrixDiagonal = A.matrixDiagonal;  //An array of pointers to the diagonal entries A.matrixValues
    const double * const rv = r.values;
    double * const xv = x.values;

    for(local_int_t i = 0; i < nrow; i++){
      const double * const currentValues = A.matrixValues[i];
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];
      const double  currentDiagonal = matrixDiagonal[i][0]; //Current diagonal value
      double sum = rv[i]; //RHS value

      for(int j = 0; j < currentNumberOfNonzeros; j++){
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j] * xv[curCol];
      }

      sum += xv[i]*currentDiagonal; //Remove diagonal contribution from previous loop
      xv[i] = sum/currentDiagonal;

    }

    //Now the back sweep.
    for(local_int_t i = nrow - 1; i >= 0; i--){
      const double * const currentValues = A.matrixValues[i];
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];
      const double  currentDiagonal = matrixDiagonal[i][0]; //Current diagonal value
      double sum = rv[i]; //RHS value

      for(int j = 0; j < currentNumberOfNonzeros; j++){
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j]*xv[curCol];
      }

      sum += xv[i]*currentDiagonal; //Remove diagonal contribution from previous loop
      xv[i] = sum/currentDiagonal;
    }
    */
    if(time != NULL) *time += mytimer();
  });
}