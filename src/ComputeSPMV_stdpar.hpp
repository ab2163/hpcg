#include "Vector.hpp"
#include "SparseMatrix.hpp"

#ifdef TIMING_ON
#include "NVTX_timing.hpp"
#endif

int ComputeSPMV_stdpar(const SparseMatrix &A, Vector  &x, Vector &y);