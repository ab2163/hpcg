#include <ranges>
#include <algorithm>
#include <execution>
#include <span>
#include <numeric>
#include <cassert>

#include "ComputeSPMV_stdpar.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

int ComputeSPMV_stdpar(const SparseMatrix &A, Vector &x, Vector &y){

  assert(x.localLength >= A.localNumberOfColumns); //test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

  #ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
  #endif

  const double * const xv = x.values;
  double * const yv = y.values;
  const double * const * const amv = A.matrixValues;
  const local_int_t * const * const indv = A.mtxIndL;
  const char * const nnz = A.nonzerosInRow;
  const local_int_t nrow = A.localNumberOfRows;
  auto rows = std::views::iota(0, nrow);

  std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [=](local_int_t i){
    double sum = 0.0;
    for(int j = 0; j < nnz[i]; j++){
      sum += amv[i][j] * xv[indv[i][j]];
    }
    yv[i] = sum;
  });

  return 0;
}