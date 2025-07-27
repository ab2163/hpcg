#include <ranges>
#include <algorithm>
#include <execution>

#include "ComputeProlongation_stdpar.hpp"
#include "NVTX_timing.hpp"

int ComputeProlongation_stdpar(const SparseMatrix &Af, Vector &xf){
  NVTX3_FUNC_RANGE();

  double * const xfv = xf.values;
  const double * const xcv = Af.mgData->xc->values;
  const local_int_t * const f2c = Af.mgData->f2cOperator;
  const local_int_t nc = Af.mgData->rc->localLength;

  auto range = std::views::iota(0, nc);

  std::for_each(std::execution::par_unseq, range.begin(), range.end(),
              [=](int i) { xfv[f2c[i]] += xcv[i]; });

  return 0;
}