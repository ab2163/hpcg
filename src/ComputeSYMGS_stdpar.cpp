#include <cassert>
#include <numeric>
#include <execution>

#include "ComputeSYMGS_ref.hpp"

int ComputeSYMGS_stdpar(const SparseMatrix & A, const Vector & r, Vector & x) {
  return ComputeSYMGS_ref(A, r, x);
}

