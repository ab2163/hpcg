#include "SparseMatrix.hpp"
#include "Vector.hpp"

#ifdef TIMING_ON
#include "NVTX_timing.hpp"
#endif

int ComputeSYMGS_par(const SparseMatrix  &A, const Vector &r, Vector &x);
