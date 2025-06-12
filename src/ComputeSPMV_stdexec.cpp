#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"

#include "ComputeSPMV_stdexec.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include <cassert>

#define NUM_THREADS 4

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
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }

  // instantiate thread pool
  exec::static_thread_pool pool(NUM_THREADS);

  // get scheduler to thread pool
  auto sched = pool.get_scheduler();

  //get sender to start from
  auto start_point = stdexec::schedule(sched);

  //bulk function to split-up workload
  auto bulk_work = stdexec::bulk(start_point, nrow, thread_spmv);

  //wait for all streams to execute
  stdexec::sync_wait(bulk_work);

  return 0;
}