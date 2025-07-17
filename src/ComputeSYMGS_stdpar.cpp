#include <cassert>
#include <numeric>
#include <execution>
#include <vector>

#include "ComputeSYMGS_ref.hpp"

int ComputeSYMGS_stdpar(const SparseMatrix & A, const Vector & r, Vector & x) {
  std::vector<int> dummyvec;
  return ComputeSYMGS_ref(A, r, x, dummyvec);
}

