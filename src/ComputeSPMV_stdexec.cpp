#include <cassert>
#include <cstdlib>

#include "ComputeSPMV_stdexec.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

template <stdexec::sender Sender>
auto ComputeSPMV_stdexec(Sender input, const SparseMatrix & A, Vector  & x, Vector & y)
  -> declype(stdexec::then(input, [](){})){

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