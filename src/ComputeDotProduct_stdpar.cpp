#include <numeric>
#include <cassert>

#include "ComputeDotProduct_stdpar.hpp"

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif

int ComputeDotProduct_stdpar(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  double local_result = 0.0;
  double * xv = x.values;
  double * yv = y.values;
  if (yv == xv) {
    local_result = std::transform_reduce(xv, xv + n, xv, 0.0); 
  }
  else {
    local_result = std::transform_reduce(xv, xv + n, yv, 0.0); 
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
