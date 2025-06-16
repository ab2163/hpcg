#include <thread>
#include <cassert>

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"
#include "ComputeDotProduct_stdexec.hpp"

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif

int ComputeDotProduct_stdexec(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  double local_result = 0.0;
  double * xv = x.values;
  double * yv = y.values;

  unsigned int num_threads = std::thread::hardware_concurrency();
  if(num_threads == 0) {
    std::cerr << "Unable to determine thread pool size.\n";
    std::exit(EXIT_FAILURE);
  }

  exec::static_thread_pool pool(num_threads);
  auto sched = pool.get_scheduler();
  auto start_point = stdexec::schedule(sched);

  if (yv == xv) {
    auto bulk_work = stdexec::bulk(start_point, n, [&](local_int_t i){ local_result += xv[i]*xv[i] });
  }
  else {
    auto bulk_work = stdexec::bulk(start_point, n, [&](local_int_t i){ local_result += xv[i]*yv[i] });
  }

  stdexec::sync_wait(bulk_work);
  
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
