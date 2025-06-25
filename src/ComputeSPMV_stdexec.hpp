#ifndef ASSERT_INCLUDED
#define ASSERT_INCLUDED
#include <cassert>
#endif

#ifndef EXECUTION_INCLUDED
#define EXECUTION_INCLUDED
#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#endif

#ifndef SPARSE_INCLUDED
#define SPARSE_INCLUDED
#include "SparseMatrix.hpp"
#endif

#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED
#include "Vector.hpp"
#endif

#ifndef TIMER_INCLUDED
#define TIMER_INCLUDED
#include "mytimer.hpp"
#endif

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

auto ComputeSPMV_stdexec(double * time, const SparseMatrix & A, Vector  & x, Vector & y){
  
  return stdexec:then([&](){
    if(time != NULL) *time -= mytimer();
    assert(x.localLength >= A.localNumberOfColumns); //Test vector lengths
    assert(y.localLength >= A.localNumberOfRows);
#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  })
  | stdexec::bulk(stdexec::par, A.localNumberOfRows, [&](local_int_t i){
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
  };)
  | stdexec::then([&](){ if(time != NULL) *time += mytimer(); });
}