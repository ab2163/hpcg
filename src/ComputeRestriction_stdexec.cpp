#include <thread>
#include <iostream>
#include <cassert>

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"
#include "ComputeRestriction_stdexec.hpp"

int ComputeRestriction_stdexec(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

  unsigned int num_threads = std::thread::hardware_concurrency();
  if(num_threads == 0) {
    std::cerr << "Unable to determine thread pool size.\n";
    std::exit(EXIT_FAILURE);
  }
  
  exec::static_thread_pool pool(num_threads);
  auto sched = pool.get_scheduler();
  auto start_point = stdexec::schedule(sched);
  auto bulk_work = stdexec::bulk(start_point, stdexec::par, nc,
    [&](int i) { rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]]; });
  stdexec::sync_wait(bulk_work);

  return 0;
}
