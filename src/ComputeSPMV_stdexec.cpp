#include <cassert>
#include <cstdlib>

#include "../stdexec/include/stdexec/execution.hpp"
#include <__senders_core.hpp>
#include "ComputeSPMV_stdexec.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

int ComputeSPMV_stdexec(stdexec::sender auto input, const SparseMatrix & A, Vector & x, Vector & y){

  assert(x.localLength >= A.localNumberOfColumns); //Test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

  auto thread_spmv = [&](local_int_t i){
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

  auto bulk_work = stdexec::bulk(start_point, stdexec::par, nrow, thread_spmv);
  return 0;
}