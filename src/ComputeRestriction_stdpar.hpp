#include "Vector.hpp"
#include "SparseMatrix.hpp"

#ifdef TIMING_ON
#include "KernelTimer.hpp"
#endif

int ComputeRestriction_stdpar(const SparseMatrix &A, const Vector &rf);