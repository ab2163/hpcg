#include "Vector.hpp"

#ifdef TIMING_ON
#include "NVTX_timing.hpp"
#endif

int ComputeDotProduct_stdpar(const local_int_t n, const Vector &x, const Vector &y,
    double &result, double &time_allreduce);