#include <cassert>
#include <cstdlib>

#include "../stdexec/include/stdexec/execution.hpp"
#include <__senders_core.hpp>
#include "ComputeSPMV_stdexec.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

int ComputeSPMV_stdexec(stdexec::sender auto input, const SparseMatrix & A, Vector & x, Vector & y){

  auto thread_spmv = [&](local_int_t i){
    const double * const xv = x.values;
    double * const yv = y.values;
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

  return input | then([&](){
    assert(x.localLength >= A.localNumberOfColumns); //Test vector lengths
    assert(y.localLength >= A.localNumberOfRows);
#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  })
  | stdexec::bulk(start_point, stdexec::par, A.localNumberOfRows, thread_spmv);
}