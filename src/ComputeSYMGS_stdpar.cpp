#include <cassert>
#include <numeric>
#include <execution>
#include <algorithm>
#include <ranges>
#include <vector>

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#include "ComputeSYMGS_ref.hpp"
#define NUM_COLORS 8
#define FWD_AND_BACK_SWEEPS 2
#define NNZ_PER_LOCN 27

int ComputeSYMGS_stdpar(const SparseMatrix &A, const Vector &r, Vector &x){
  assert(x.localLength == A.localNumberOfColumns);
#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  const double * const * const matrixDiagonal = A.matrixDiagonal;  //array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;
  const double * const * const amv = A.matrixValues;
  const local_int_t * const * const indv = A.mtxIndL;
  const char * const nnz = A.nonzerosInRow;
  const unsigned char * const colors = A.colors;

  const local_int_t * const mem2row = A.mem2row;
  const double * const startOfMemVals = A.startOfMemVals;
  const local_int_t * const startOfMemInds = A.startOfMemInds;

  for(int sweeps = 0; sweeps < FWD_AND_BACK_SWEEPS; sweeps++){
    for(int color = 0; color < NUM_COLORS; color++){

      //work over a smaller range for the specific colour
      auto locns = std::views::iota(A.startInds[color], A.endInds[color]);

      std::for_each(std::execution::par_unseq, locns.begin(), locns.end(), [=](local_int_t i){
        local_int_t ind = mem2row[i];
        const double currentDiagonal = matrixDiagonal[ind][0]; //current diagonal value
        double sum = rv[ind]; //RHS value

        for(int j = 0; j < nnz[ind]; j++){
          local_int_t curCol = startOfMemInds[NNZ_PER_LOCN*i + j];
          sum -= startOfMemVals[NNZ_PER_LOCN*i + j] * xv[curCol];
        }
        sum += xv[ind]*currentDiagonal; //remove diagonal contribution from previous loop

        xv[ind] = sum/currentDiagonal;
      });
    }
  }

  return 0;
}

