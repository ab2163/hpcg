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
#include <cmath>

#ifdef USE_GPU
#include <nvexec/stream_context.cuh>
#endif

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
#define NUM_BINS 1000

#ifndef HPCG_NO_MPI
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([&](){ \
    MPI_Allreduce(&local_result, &(RESULT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); \
  })
#else
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  bulk(stdexec::par_unseq, NUM_BINS, [=](local_int_t i){ \
    local_int_t minInd = i*(nrow/NUM_BINS); \
    local_int_t maxInd = ((i + 1) == NUM_BINS) ? nrow : (i + 1)*(nrow/NUM_BINS); \
    double bin_sum = 0.0; \
    for(local_int_t j = minInd; j < maxInd; ++j) { \
      bin_sum = std::fma((VEC1VALS)[j], (VEC2VALS)[j], bin_sum); \
    } \
    bin_vals[i] = bin_sum; \
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
    if((COLORS)[i] == 0){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 1){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 2){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 3){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 4){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 5){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 6){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 7){ \
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
    | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 0){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 1){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 2){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 3){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 4){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 5){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 6){ \
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
  | bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == 7){ \
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

#define SYMGS(DEPTH) \
  if((DEPTH) == 0){ \
    sync_wait(symgs_sweep_0); \
  }else if((DEPTH) == 1){ \
    sync_wait(symgs_sweep_1); \
  }else if((DEPTH) == 2){ \
    sync_wait(symgs_sweep_2); \
  }else if((DEPTH) == 3){ \
    sync_wait(symgs_sweep_3); \
  } \

#define SYMGS_SWEEP_0() \
  SYMGS_SWEEP(A_vals_const[0], z_vals[0], r_vals[0], A_nnzs_const[0], A_inds_const[0], A_nrows_const[0], A_diags_const[0], A_colors_const[0])

#define SYMGS_SWEEP_1() \
  SYMGS_SWEEP(A_vals_const[1], z_vals[1], r_vals[1], A_nnzs_const[1], A_inds_const[1], A_nrows_const[1], A_diags_const[1], A_colors_const[1])

#define SYMGS_SWEEP_2() \
  SYMGS_SWEEP(A_vals_const[2], z_vals[2], r_vals[2], A_nnzs_const[2], A_inds_const[2], A_nrows_const[2], A_diags_const[2], A_colors_const[2])

#define SYMGS_SWEEP_3() \
  SYMGS_SWEEP(A_vals_const[3], z_vals[3], r_vals[3], A_nnzs_const[3], A_inds_const[3], A_nrows_const[3], A_diags_const[3], A_colors_const[3])

#define MGP0a() \
  ZeroVector(*z_objs[0]);
#define MGP0b() \
  SYMGS(0)
#define MGP0c() \
  SPMV(A_vals_const[0], z_vals[0], Axfv_vals[0], A_inds_const[0], A_nnzs_const[0], A_nrows_const[0]) \
  | RESTRICTION(*A_objs[0], 0)

#define MGP1a() \
  ZeroVector(*z_objs[1]);
#define MGP1b() \
  SYMGS(1)
#define MGP1c() \
  SPMV(A_vals_const[1], z_vals[1], Axfv_vals[1], A_inds_const[1], A_nnzs_const[1], A_nrows_const[1]) \
  | RESTRICTION(*A_objs[1], 1)

#define MGP2a() \
  ZeroVector(*z_objs[2]);
#define MGP2b() \
  SYMGS(2)
#define MGP2c() \  
  SPMV(A_vals_const[2], z_vals[2], Axfv_vals[2], A_inds_const[2], A_nnzs_const[2], A_nrows_const[2]) \
  | RESTRICTION(*A_objs[2], 2)

#define MGP3a() \
  ZeroVector(*z_objs[3]);
#define MGP3b() \
  SYMGS(3)

#define MGP4a() \
  PROLONGATION(*A_objs[2], 2)
#define MGP4b() \
  SYMGS(2)

#define MGP5a() \
  PROLONGATION(*A_objs[1], 1)
#define MGP5b() \
  SYMGS(1)

#define MGP6a() \
  PROLONGATION(*A_objs[0], 0)
#define MGP6b() \
  SYMGS(0)

int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
  const int max_iter, const double tolerance, int &niters, double &normr,  double &normr0,
  double *times, bool doPreconditioning);