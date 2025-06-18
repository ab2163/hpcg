#include <ranges>
#include <algorithm>
#include <execution>

#include "ComputeRestriction_stdpar.hpp"

int ComputeRestriction_stdpar(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

  auto range = std::views::iota(0, nc);

  std::for_each(std::execution::par, range.begin(), range.end(),
              [&](int i) { rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]]; });
              
  return 0;
}
