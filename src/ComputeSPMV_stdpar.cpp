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
  const local_int_t nrow = A.localNumberOfRows;
  auto rows = std::views::iota(local_int_t{0}, nrow);

  std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [&, xv, yv](local_int_t i){
    double sum = 0.0;
    double *amv = A.matrixValues[i];
    local_int_t *indv = A.mtxIndL[i];
    for(int j = 0; j < A.nonzerosInRow[i]; ++j){
      sum += amv[j] * xv[indv[j]];
    }
    yv[i] = sum;
  });

  return 0;
}