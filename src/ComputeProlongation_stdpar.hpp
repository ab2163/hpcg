#include "Vector.hpp"
#include "SparseMatrix.hpp"

#ifdef TIMING_ON
#include "NVTX_timing.hpp"
#endif

int ComputeProlongation_stdpar(const SparseMatrix &Af, Vector &xf);