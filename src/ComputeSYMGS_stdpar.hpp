#include "SparseMatrix.hpp"
#include "Vector.hpp"

#ifdef TIMING_ON
#include "KernelTimer.hpp"
#endif

int ComputeSYMGS_stdpar(const SparseMatrix  &A, const Vector &r, Vector &x);