#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>
#include <numeric>
#include <iostream>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include "ComputeSYMGS_ref.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "mytimer.hpp"
#include "hpcg.hpp"

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

using stdexec::sender;
using stdexec::then;
using stdexec::schedule;
using stdexec::sync_wait;
using stdexec::bulk;
using stdexec::just;
using stdexec::continues_on;

#define NUM_MG_LEVELS 4
#define SINGLE_THREAD 1

#ifndef HPCG_NO_MPI
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([&](){ \
    local_result = 0.0; \
    local_result = std::transform_reduce(std::execution::par_unseq, (VEC1VALS), (VEC1VALS) + nrow, (VEC2VALS), 0.0); \
    MPI_Allreduce(&local_result, &(RESULT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); \
  })
#else
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([&](){ \
    (RESULT) = std::transform_reduce(std::execution::par_unseq, (VEC1VALS), (VEC1VALS) + nrow, (VEC2VALS), 0.0); \
  })
#endif

#define WAXPBY(ALPHA, XVALS, BETA, YVALS, WVALS) \
  bulk(stdexec::par_unseq, nrow, [&](local_int_t i){ (WVALS)[i] = (ALPHA)*(XVALS)[i] + (BETA)*(YVALS)[i]; })

#ifndef HPCG_NO_MPI
#define SPMV(A, x, y) \
  then([&](){ ExchangeHalo((A), (x)); }) \
  | bulk(stdexec::par_unseq, (A).localNumberOfRows, [=](local_int_t i){ \
    double sum = 0.0; \
    const double * const cur_vals = (A).matrixValues[i]; \
    const local_int_t * const cur_inds = (A).mtxIndL[i]; \
    const int cur_nnz = (A).nonzerosInRow[i]; \
    const double * const xv = (x).values; \
    double * const yv = (y).values; \
    for(int j = 0; j < cur_nnz; j++) \
      sum += cur_vals[j]*xv[cur_inds[j]]; \
    yv[i] = sum; \
  })
#else
#define SPMV(A, x, y) \
  bulk(stdexec::par_unseq, (A).localNumberOfRows, [=](local_int_t i){ \
    double sum = 0.0; \
    const double * const cur_vals = (A).matrixValues[i]; \
    const local_int_t * const cur_inds = (A).mtxIndL[i]; \
    const int cur_nnz = (A).nonzerosInRow[i]; \
    const double * const xv = (x).values; \
    double * const yv = (y).values; \
    for(int j = 0; j < cur_nnz; j++) \
      sum += cur_vals[j]*xv[cur_inds[j]]; \
    yv[i] = sum; \
  })
#endif

#define RESTRICTION(A, rf, level) \
  bulk(stdexec::par_unseq, (A).mgData->rc->localLength, \
    [&](int i){ \
    rcv_ptrs[(level)][i] = rfv_ptrs[(level)][f2c_ptrs[(level)][i]] - Axfv_ptrs[(level)][f2c_ptrs[(level)][i]]; \
  })

#define PROLONGATION(Af, xf, level) \
  bulk(stdexec::par_unseq, (Af).mgData->rc->localLength, \
    [&](int i){ \
    xfv_ptrs[(level)][f2c_ptrs[(level)][i]] += xcv_ptrs[(level)][i]; \
  })

//NOTE - OMITTED MPI HALOEXCHANGE IN SYMGS
#define PRE_RECURSION_MG(A, r, x, level) \
  continues_on(scheduler_single_thread) \
  | then([&](){ \
    ZeroVector((x)); \
    ComputeSYMGS_ref((A), (r), (x)); \
  }) \
  | continues_on(scheduler) \
  | SPMV((A), (x), *((A).mgData->Axf)) \
  | RESTRICTION((A), (r), (level)) \

#define POST_RECURSION_MG(A, r, x, level) \
  PROLONGATION((A), (x), (level)) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ \
    ComputeSYMGS_ref((A), (r), (x)); \
  }) \
  | continues_on(scheduler)

#define TERMINAL_MG(A, r, x) \
  continues_on(scheduler_single_thread) \
  | then([&](){ \
    ZeroVector((x)); \
    ComputeSYMGS_ref((A), (r), (x)); \
  }) \
  | continues_on(scheduler)
  
#define COMPUTE_MG_STAGE1() \
  PRE_RECURSION_MG(A0, r0, z0, 0) \
  | PRE_RECURSION_MG(A1, r1, z1, 1) \
  | PRE_RECURSION_MG(A2, r2, z2, 2)

#define COMPUTE_MG_STAGE2() \
  TERMINAL_MG(A3, r3, z3) \
  | POST_RECURSION_MG(A2, r2, z2, 2) \
  | POST_RECURSION_MG(A1, r1, z1, 1) \
  | POST_RECURSION_MG(A0, r0, z0, 0)

int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
  const int max_iter, const double tolerance, int &niters, double &normr,  double &normr0,
  double *times, bool doPreconditioning);