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
#include <omp.h>

#ifdef USE_GPU
#include <nvexec/stream_context.cuh>
#endif

#include "ComputeSYMGS_ref.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "mytimer.hpp"
#include "hpcg.hpp"
#include "NVTX_timing.hpp"

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
#define NUM_BINS 1000

#define TW(TASK, MESSAGE) \
  start_timing(MESSAGE, rangeID); \
  sync_wait(schedule(scheduler) | TASK); \
  end_timing(rangeID);

#ifndef HPCG_NO_MPI
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([&](){ \
    MPI_Allreduce(&local_result, &(RESULT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); \
  })
#else
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  bulk(stdexec::par_unseq, nrow, [=](local_int_t i){ prod_vals[i] = (VEC1VALS)[i]*(VEC2VALS)[i]; }) \
  | bulk(stdexec::par_unseq, NUM_BINS, [=](local_int_t i){ \
    local_int_t minInd = i*(nrow/NUM_BINS); \
    local_int_t maxInd; \
    double val_cpy = 0.0; \
    if((i + 1) == NUM_BINS) maxInd = nrow; \
    else maxInd = (i + 1)*(nrow/NUM_BINS); \
    for(local_int_t j = minInd; j < maxInd; j++){ \
      val_cpy += prod_vals[j]; \
    } \
    bin_vals[i] = val_cpy; \
  }) \
  | then([=](){ \
    double result_cpy = 0.0; \
    for(local_int_t i = 0; i < NUM_BINS; i++) result_cpy += bin_vals[i]; \
    *(RESULT) = result_cpy; \
  })
#endif

#define WAXPBY(ALPHA, XVALS, BETA, YVALS, WVALS) \
  bulk(stdexec::par_unseq, nrow, [=](local_int_t i){ (WVALS)[i] = (ALPHA)*(XVALS)[i] + (BETA)*(YVALS)[i]; })

//CURRENTLY IGNORING HALO EXCHANGE WITH SPMV
#define SPMV(AMV, XV, YV, INDV, NNZ, NROW) \
  bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    double sum = 0.0; \
    for(int j = 0; j < (NNZ)[i]; j++){ \
      sum += (AMV)[i][j] * (XV)[(INDV)[i][j]]; \
    } \
    (YV)[i] = sum; \
  }) \

#define RESTRICTION(A, depth) \
  bulk(stdexec::par_unseq, (A).mgData->rc->localLength, [=](int i){ \
      rcv_vals[(depth)][i] = r_vals[(depth)][f2c_vals[(depth)][i]] - Axfv_vals[(depth)][f2c_vals[(depth)][i]]; \
  })

#define PROLONGATION(Af, depth) \
  bulk(stdexec::par_unseq, (Af).mgData->rc->localLength, [=](int i){ \
    z_vals[(depth)][f2c_vals[(depth)][i]] += xcv_vals[(depth)][i]; \
  })

//NOTE - OMITTED MPI HALOEXCHANGE IN SYMGS
#define SYMGS_SWEEP(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \
  bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == *color){ \
        const double currentDiagonal = (MATR_DIAG)[i][0]; \
        double sum = (RVALS)[i]; \
        for(int j = 0; j < (NNZ)[i]; j++){ \
          local_int_t curCol = (INDV)[i][j]; \
          sum -= (AMV)[i][j] * (XVALS)[curCol]; \
        } \
        sum += (XVALS)[i]*currentDiagonal; \
        (XVALS)[i] = sum/currentDiagonal; \
      } \
  }) \
  | then([=](){ (*color)++; }) \
  | continues_on(scheduler_cpu) \
  | repeat_n(NUM_COLORS)

#define SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \
  SYMGS_SWEEP(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \   
  | then([=](){ *color = 0; }) \
  | repeat_n(FORWARD_AND_BACKWARD) \

#define MGP0a() \
  sync_wait(schedule(scheduler_cpu) | then([&](){ ZeroVector(*z_objs[0]); }));
#define MGP0b() \
  TW(SYMGS(A_vals[0], z_vals[0], r_vals[0], A_nnzs[0], A_inds[0], A_nrows[0], A_diags[0], A_colors[0]), "SYMGS")
#define MGP0c() \
  TW(SPMV(A_vals[0], z_vals[0], Axfv_vals[0], A_inds[0], A_nnzs[0], A_nrows[0]), "SPMV") \
  TW(RESTRICTION(*A_objs[0], 0), "Restriction")

#define MGP1a() \
  sync_wait(schedule(scheduler_cpu) | then([&](){ ZeroVector(*z_objs[1]); }));
#define MGP1b() \
  TW(SYMGS(A_vals[1], z_vals[1], r_vals[1], A_nnzs[1], A_inds[1], A_nrows[1], A_diags[1], A_colors[1]), "SYMGS")
#define MGP1c() \
  TW(SPMV(A_vals[1], z_vals[1], Axfv_vals[1], A_inds[1], A_nnzs[1], A_nrows[1]), "SPMV") \
  TW(RESTRICTION(*A_objs[1], 1), "Restriction")

#define MGP2a() \
  sync_wait(schedule(scheduler_cpu) | then([&](){ ZeroVector(*z_objs[2]); }));
#define MGP2b() \
  TW(SYMGS(A_vals[2], z_vals[2], r_vals[2], A_nnzs[2], A_inds[2], A_nrows[2], A_diags[2], A_colors[2]), "SYMGS")
#define MGP2c() \  
  TW(SPMV(A_vals[2], z_vals[2], Axfv_vals[2], A_inds[2], A_nnzs[2], A_nrows[2]), "SPMV") \
  TW(RESTRICTION(*A_objs[2], 2), "Restriction")

#define MGP3a() \
  sync_wait(schedule(scheduler_cpu) | then([&](){ ZeroVector(*z_objs[3]); }));
#define MGP3b() \
  TW(SYMGS(A_vals[3], z_vals[3], r_vals[3], A_nnzs[3], A_inds[3], A_nrows[3], A_diags[3], A_colors[3]), "SYMGS")

#define MGP4a() \
  TW(PROLONGATION(*A_objs[2], 2), "Prolongation")
#define MGP4b() \
  TW(SYMGS(A_vals[2], z_vals[2], r_vals[2], A_nnzs[2], A_inds[2], A_nrows[2], A_diags[2], A_colors[2]), "SYMGS")

#define MGP5a() \
  TW(PROLONGATION(*A_objs[1], 1), "Prolongation")
#define MGP5b() \
  TW(SYMGS(A_vals[1], z_vals[1], r_vals[1], A_nnzs[1], A_inds[1], A_nrows[1], A_diags[1], A_colors[1]), "SYMGS")

#define MGP6a() \
  TW(PROLONGATION(*A_objs[0], 0), "Prolongation")
#define MGP6b() \
  TW(SYMGS(A_vals[0], z_vals[0], r_vals[0], A_nnzs[0], A_inds[0], A_nrows[0], A_diags[0], A_colors[0]), "SYMGS")

int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
  const int max_iter, const double tolerance, int &niters, double &normr,  double &normr0,
  double *times, bool doPreconditioning);