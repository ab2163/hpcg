//CURRENTLY SAME AS REFERENCE IMPLEMENTATION
//CHANGE THIS CODE!

#include "ComputeWAXPBY_stdexec.hpp"
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

int ComputeWAXPBY_stdexec(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  const double * const xv = x.values;
  const double * const yv = y.values;
  double * const wv = w.values;

  if (alpha==1.0) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (local_int_t i=0; i<n; i++) wv[i] = xv[i] + beta * yv[i];
  } else if (beta==1.0) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + yv[i];
  } else  {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + beta * yv[i];
  }

  return 0;
}
