#include "Vector.hpp"
#include "SparseMatrix.hpp"

#ifdef TIMING_ON
#include "NVTX_timing.hpp"
#endif

int ComputeRestriction_stdpar(const SparseMatrix &A, const Vector &rf);