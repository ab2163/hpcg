#include <ranges>
#include <algorithm>
#include <execution>

#include "ComputeRestriction_stdpar.hpp"

int ComputeRestriction_stdpar(const SparseMatrix &A, const Vector &rf){

  const double * const Axfv = A.mgData->Axf->values;
  const double * const rfv = rf.values;
  double * const rcv = A.mgData->rc->values;
  const local_int_t * const f2c = A.mgData->f2cOperator;
  const local_int_t nc = A.mgData->rc->localLength;

  auto range = std::views::iota(0, nc);

  std::for_each(std::execution::par_unseq, range.begin(), range.end(),
              [=](int i) { rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]]; });
  
  return 0;
}
