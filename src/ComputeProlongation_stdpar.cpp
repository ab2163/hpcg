#include <ranges>
#include <algorithm>
#include <execution>

#include "ComputeProlongation_stdpar.hpp"

int ComputeProlongation_stdpar(const SparseMatrix &Af, Vector &xf){

  double * xfv = xf.values;
  double * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

  auto range = std::views::iota(0, nc);

  std::for_each(std::execution::par_unseq, range.begin(), range.end(),
              [&](int i) { xfv[f2c[i]] += xcv[i]; });

  return 0;
}