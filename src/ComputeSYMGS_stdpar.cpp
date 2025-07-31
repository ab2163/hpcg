#include <cassert>
#include <numeric>
#include <execution>
#include <algorithm>
#include <ranges>
#include <vector>

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#include "ComputeSYMGS_stdpar.hpp"
#define NUM_COLORS 8

int ComputeSYMGS_stdpar(const SparseMatrix &A, const Vector &r, Vector &x){

#ifdef TIMING_ON
  GlobalKernelTimer.start(SYMGS);
#endif

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
  auto rows = std::views::iota(local_int_t{0}, nrow);

  for(int color = 0; color < NUM_COLORS; color++){
    std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [=](local_int_t i){
      if(colors[i] == color){
        const double currentDiagonal = matrixDiagonal[i][0]; //current diagonal value
        double sum = rv[i]; //RHS value

        for(int j = 0; j < nnz[i]; j++){
          local_int_t curCol = indv[i][j];
          sum -= amv[i][j] * xv[curCol];
        }
        sum += xv[i]*currentDiagonal; //remove diagonal contribution from previous loop

        xv[i] = sum/currentDiagonal;
      }
    });
  }

  //back sweep
  for(int color = 0; color < NUM_COLORS; color++){
    std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [=](local_int_t i){
      if(colors[i] == color){
        const double currentDiagonal = matrixDiagonal[i][0]; //current diagonal value
        double sum = rv[i]; //RHS value

        for(int j = 0; j < nnz[i]; j++){
          local_int_t curCol = indv[i][j];
          sum -= amv[i][j] * xv[curCol];
        }
        sum += xv[i]*currentDiagonal; //remove diagonal contribution from previous loop

        xv[i] = sum/currentDiagonal;
      }
    });
  }

#ifdef TIMING_ON
  GlobalKernelTimer.stop(SYMGS);
#endif

  return 0;
}

