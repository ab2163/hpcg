#include <thread>
#include <iostream>
#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"

#include "ComputeSPMV_stdexec.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include <cassert>

int ComputeSPMV_stdexec(const SparseMatrix & A, Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

  auto thread_spmv = [&](local_int_t i) {
    yv[i] = std::transform_reduce(
      A.matrixValues[i],
      A.matrixValues[i] + A.nonzerosInRow[i],
      std::views::transform(
        std::span(A.mtxIndL[i], A.nonzerosInRow[i]),
        [&xv](int idx) { return xv[idx]; }
      ).begin(),
      0.0
    );
  };

  unsigned int num_threads = std::thread::hardware_concurrency();
  if(num_threads == 0) {
    std::cerr << "Unable to determine thread pool size.\n";
    std::exit(EXIT_FAILURE);
  }

  exec::static_thread_pool pool(num_threads);
  auto sched = pool.get_scheduler();
  auto start_point = stdexec::schedule(sched);
  auto bulk_work = stdexec::bulk(start_point, nrow, thread_spmv);
  stdexec::sync_wait(bulk_work);
  return 0;
}