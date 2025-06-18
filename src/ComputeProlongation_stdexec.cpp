#include <thread>
#include <iostream>
#include <cassert>

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"
#include "ComputeProlongation_stdexec.hpp"

int ComputeProlongation_stdexec(const SparseMatrix & Af, Vector & xf) {

  double * xfv = xf.values;
  double * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

  unsigned int num_threads = std::thread::hardware_concurrency();
  if(num_threads == 0) {
    std::cerr << "Unable to determine thread pool size.\n";
    std::exit(EXIT_FAILURE);
  }

  exec::static_thread_pool pool(num_threads);
  auto sched = pool.get_scheduler();
  auto start_point = stdexec::schedule(sched);
  auto bulk_work = stdexec::bulk(start_point, stdexec::par, nc,
    [&](int i) { xfv[f2c[i]] += xcv[i]; });
  stdexec::sync_wait(bulk_work);

  return 0;
}