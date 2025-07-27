#include <cassert>
#include <numeric>
#include <execution>

#include "ComputeSYMGS_ref.hpp"
#include "NVTX_timing.hpp"

int ComputeSYMGS_stdpar(const SparseMatrix &A, const Vector &r, Vector &x){
  NVTX3_FUNC_RANGE();
  return ComputeSYMGS_ref(A, r, x);
}

