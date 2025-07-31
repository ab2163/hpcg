#include "Vector.hpp"
#include "SparseMatrix.hpp"

#ifdef TIMING_ON
#include "KernelTimer.hpp"
#endif

int ComputeSPMV_stdpar(const SparseMatrix &A, Vector  &x, Vector &y);