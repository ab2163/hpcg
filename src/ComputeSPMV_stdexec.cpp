#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"

#include "ComputeSPMV_stdexec.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include <cassert>

#define NUM_THREADS 4

using stdexec::starts_on;
using stdexec::just;
using stdexec::when_all;
using stdexec::sync_wait;

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

  local_int_t base = nrow/NUM_THREADS;
  int remainder = nrow % NUM_THREADS;

  for(cnt = 0; cnt < NUM_THREADS; cnt++) {
    if(cnt < remainder)
      starts_on(sched, just(thread_spmv(0, 2, 500)));
    else
      starts_on(sched, just(thread_spmv(0, 2, 500)));
  }

  return 0;
}