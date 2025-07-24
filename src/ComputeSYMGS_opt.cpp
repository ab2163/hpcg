#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include "ComputeSYMGS_ref.hpp"
#include <cassert>
#include "NVTX_timing.hpp"
#define NUM_COLORS 8

int ComputeSYMGS_opt(const SparseMatrix &A, const Vector &r, Vector &x){
  nvtxRangeId_t rangeID = 0;
  start_timing("SYMGS_opt", rangeID);
  assert(x.localLength == A.localNumberOfColumns);
#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

  const double * const rv = r.values;
  double * const xv = x.values;

  for(int color = 0; color < NUM_COLORS; color++){
    local_int_t startInd = A.startInds[color];
    local_int_t endInd = A.endInds[color];

    #pragma omp parallel for
    for(local_int_t ind = startInd; ind <= endInd; ind++){
      RowDataFlat *rowPtr = &A.rowStructs[ind];
      double *vals = rowPtr->values;
      double **xVals = rowPtr->xVals;
      int nnz = rowPtr->numNonzeros;
      double diagVal = rowPtr->diagVal;
      int i = rowPtr->rowIndex;
      double sum = rv[i]; //RHS value

      for (int j = 0; j < nnz; j++){
        sum -= vals[j]*(*xVals[j]);
      }
      sum += xv[i]*diagVal; //remove diagonal contribution from previous loop
      xv[i] = sum/diagVal;
    }
  }

  //back sweep
  for(int color = NUM_COLORS - 1; color >= 0; color--){
    local_int_t startInd = A.startInds[color];
    local_int_t endInd = A.endInds[color];

    #pragma omp parallel for
    for(local_int_t ind = startInd; ind <= endInd; ind++){
      RowDataFlat *rowPtr = &A.rowStructs[ind];
      double *vals = rowPtr->values;
      double **xVals = rowPtr->xVals;
      int nnz = rowPtr->numNonzeros;
      double diagVal = rowPtr->diagVal;
      int i = rowPtr->rowIndex;
      double sum = rv[i]; //RHS value

      for(int j = 0; j < nnz; j++){
        sum -= vals[j]*(*xVals[j]);
      }
      sum += xv[i]*diagVal; //remove diagonal contribution from previous loop
      xv[i] = sum/diagVal;
    }
  }

  end_timing(rangeID);
  return 0;
}

