#include <ranges>
#include <algorithm>
#include <execution>
#include <span>
#include <numeric>

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

  // create a lazy-evaluated view
  auto indices = std::ranges::views::iota(static_cast<local_int_t>(0), nrow);

  // run parallel for with the lazy view
  std::for_each(std::execution::par, indices.begin(), indices.end(), [&](local_int_t i) {
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    auto xv_view = std::views::transform(
      std::span(cur_inds, cur_nnz),
      [&xv](int idx) { return xv[idx]; }
    );

  yv[i] = std::transform_reduce(cur_vals, cur_vals+cur_nnz, xv_view.begin(), 0);
  });
  return 0;
}