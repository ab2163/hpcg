#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>
#include <numeric>
#include <iostream>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#include <exec/repeat_n.hpp>
#include <nvexec/stream_context.cuh>

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
using exec::repeat_n;

#define NUM_MG_LEVELS 4
#define NUM_COLORS 8
#define FORWARD_AND_BACKWARD 2

#ifndef HPCG_NO_MPI
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([&](){ \
    local_result = 0.0; \
    local_result = std::transform_reduce(std::execution::par_unseq, (VEC1VALS), (VEC1VALS) + nrow, (VEC2VALS), 0.0); \
    MPI_Allreduce(&local_result, &(RESULT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); \
  })
#else
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([=](){ \
  })
#endif

#define WAXPBY(ALPHA, XVALS, BETA, YVALS, WVALS) \
  bulk(stdexec::par_unseq, nrow, [=](local_int_t i){ })

//CURRENTLY IGNORING HALO EXCHANGE WITH SPMV
#define SPMV(AMV, XV, YV, INDV, NNZ, NROW) \
  bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ })

#define RESTRICTION(A, depth) \
  bulk(stdexec::par_unseq, (A).mgData->rc->localLength, \
    [=](int i){ })

#define PROLONGATION(Af, depth) \
  bulk(stdexec::par_unseq, (Af).mgData->rc->localLength, \
    [=](int i){ })

//NOTE - OMITTED MPI HALOEXCHANGE IN SYMGS
#define SYMGS_SWEEP(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \
  bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ }) \
  | then([=](){ (*color)++; }) \
  | continues_on(scheduler_cpu) \
  | repeat_n(NUM_COLORS)

#define SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \
  SYMGS_SWEEP(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \   
  | then([=](){ *color = 0; }) \
  | repeat_n(FORWARD_AND_BACKWARD) \

#define MGP0a() \
  then([&](){ /*ZeroVector(*z_objs[0]);*/ })
#define MGP0b() \
  SYMGS(A_vals[0], z_vals[0], r_vals[0], A_nnzs[0], A_inds[0], A_nrows[0], A_diags[0], A_colors[0])
#define MGP0c() \
  SPMV(A_vals[0], z_vals[0], Axfv_vals[0], A_inds[0], A_nnzs[0], A_nrows[0]) \
  | RESTRICTION(*A_objs[0], 0)

#define MGP1a() \
  then([&](){ /*ZeroVector(*z_objs[1]);*/ })
#define MGP1b() \
  SYMGS(A_vals[1], z_vals[1], r_vals[1], A_nnzs[1], A_inds[1], A_nrows[1], A_diags[1], A_colors[1])
#define MGP1c() \
  SPMV(A_vals[1], z_vals[1], Axfv_vals[1], A_inds[1], A_nnzs[1], A_nrows[1]) \
  | RESTRICTION(*A_objs[1], 1)

#define MGP2a() \
  then([&](){ /*ZeroVector(*z_objs[2]);*/ })
#define MGP2b() \
  SYMGS(A_vals[2], z_vals[2], r_vals[2], A_nnzs[2], A_inds[2], A_nrows[2], A_diags[2], A_colors[2])
#define MGP2c() \  
  SPMV(A_vals[2], z_vals[2], Axfv_vals[2], A_inds[2], A_nnzs[2], A_nrows[2]) \
  | RESTRICTION(*A_objs[2], 2)

#define MGP3a() \
  then([&](){ /*ZeroVector(*z_objs[3]);*/ })
#define MGP3b() \
  SYMGS(A_vals[3], z_vals[3], r_vals[3], A_nnzs[3], A_inds[3], A_nrows[3], A_diags[3], A_colors[3])

#define MGP4a() \
  PROLONGATION(*A_objs[2], 2)
#define MGP4b() \
  SYMGS(A_vals[2], z_vals[2], r_vals[2], A_nnzs[2], A_inds[2], A_nrows[2], A_diags[2], A_colors[2])

#define MGP5a() \
  PROLONGATION(*A_objs[1], 1)
#define MGP5b() \
  SYMGS(A_vals[1], z_vals[1], r_vals[1], A_nnzs[1], A_inds[1], A_nrows[1], A_diags[1], A_colors[1])

#define MGP6a() \
  PROLONGATION(*A_objs[0], 0)
#define MGP6b() \
  SYMGS(A_vals[0], z_vals[0], r_vals[0], A_nnzs[0], A_inds[0], A_nrows[0], A_diags[0], A_colors[0])

int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
  const int max_iter, const double tolerance, int &niters, double &normr,  double &normr0,
  double *times, bool doPreconditioning);