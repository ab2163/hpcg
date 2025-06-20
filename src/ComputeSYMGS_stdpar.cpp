#include <cassert>
#include <numeric>
#include <execution>

#include "ComputeSYMGS_stdpar.hpp"

int ComputeSYMGS_stdpar( const SparseMatrix & A, const Vector & r, Vector & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;

  for (local_int_t i=0; i< nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    sum -= std::transform_reduce(std::execution::par, 
      currentValues, 
      currentValues + currentNumberOfNonzeros,
      currentColIndices,
      0.0,
      std::plus<>(),
      [&](double matrixVal, int ind) { return matrixVal*xv[ind]; }
    );

    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;

  }

  // Now the back sweep.

  for (local_int_t i=nrow-1; i>=0; i--) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    sum -= std::transform_reduce(std::execution::par, 
      currentValues, 
      currentValues + currentNumberOfNonzeros,
      currentColIndices,
      0.0,
      std::plus<>(),
      [&](double matrixVal, int ind) { return matrixVal*xv[ind]; }
    );

    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
  }

  return 0;
}

