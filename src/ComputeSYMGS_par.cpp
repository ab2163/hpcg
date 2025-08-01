#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#include <cassert>
#include "ComputeSYMGS_par.hpp"
#define NUM_COLORS 8

int ComputeSYMGS_par(const SparseMatrix &A, const Vector &r, Vector &x){
  assert(x.localLength == A.localNumberOfColumns);
#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  const double * const * const matrixDiagonal = A.matrixDiagonal;  //array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;

  for(int color = 0; color < NUM_COLORS; color++){
    #pragma omp parallel for
    for(local_int_t i = 0; i < nrow; i++){
      if(A.colors[i] != color) continue;
      const double * const currentValues = A.matrixValues[i];
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];
      const double currentDiagonal = matrixDiagonal[i][0]; //current diagonal value
      double sum = rv[i]; //RHS value

      for(int j = 0; j < currentNumberOfNonzeros; j++){
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j] * xv[curCol];
      }
      sum += xv[i]*currentDiagonal; //remove diagonal contribution from previous loop

      xv[i] = sum/currentDiagonal;
    }
  }

  //back sweep
  for(int color = 0; color < NUM_COLORS; color++){
    #pragma omp parallel for
    for(local_int_t i = nrow - 1; i >= 0; i--){
      if(A.colors[i] != color) continue;
      const double * const currentValues = A.matrixValues[i];
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];
      const double  currentDiagonal = matrixDiagonal[i][0]; //current diagonal value
      double sum = rv[i]; //RHS value

      for(int j = 0; j < currentNumberOfNonzeros; j++){
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j]*xv[curCol];
      }
      sum += xv[i]*currentDiagonal; //remove diagonal contribution from previous loop

      xv[i] = sum/currentDiagonal;
    }
  }

  return 0;
}

