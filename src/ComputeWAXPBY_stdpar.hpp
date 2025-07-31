#include "Vector.hpp"

#ifdef TIMING_ON
#include "KernelTimer.hpp"
#endif

int ComputeWAXPBY_stdpar(const local_int_t n, const double alpha, const Vector &x,
    const double beta, const Vector &y, Vector &w);