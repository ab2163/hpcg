#include <ranges>
#include <algorithm>
#include <execution>
#include <span>

#include "ComputeSPMV_stdpar.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include <cassert>

int ComputeSPMV_stdpar(const SparseMatrix & A, Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

  #ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
  #endif

  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
  auto rows = std::views::iota(0, nrow);

  // Create a view for xv values with "transform"
  // Pass the view and the A values to "transform reduce"
  std::for_each(std::execution::par, rows.begin(), rows.end(), [&](local_int_t i) {
    yv[i] = std::transform_reduce(
      A.matrixValues[i], 
      A.matrixValues[i] + A.nonzerosInRow[i],
      std::views::transform(
        std::span(A.mtxIndL[i], A.nonzerosInRow[i]),
        [xv](int idx) { return xv[idx]; }
      ).begin(),
      0.0
    );
  });

  return 0;
}