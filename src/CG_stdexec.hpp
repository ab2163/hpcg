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
#define SINGLE_THREAD 1
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
  then([&](){ \
    (RESULT) = std::transform_reduce(std::execution::par_unseq, (VEC1VALS), (VEC1VALS) + nrow, (VEC2VALS), 0.0); \
  })
#endif

#define WAXPBY(ALPHA, XVALS, BETA, YVALS, WVALS) \
  bulk(stdexec::par_unseq, nrow, [&](local_int_t i){ (WVALS)[i] = (ALPHA)*(XVALS)[i] + (BETA)*(YVALS)[i]; })

//CURRENTLY IGNORING HALO EXCHANGE WITH SPMV
#define SPMV(AMV, XV, YV, INDV, NNZ, NROW) \
  bulk(stdexec::par_unseq, (NROW), [&](local_int_t i){ \
    double sum = 0.0; \
    for(int j = 0; j < (NNZ)[i]; j++){ \
      sum += (AMV)[i][j] * (XV)[(INDV)[i][j]]; \
    } \
    (YV)[i] = sum; \
  }) \

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
#define SYMGS_SWEEP(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \
  bulk(stdexec::par_unseq, (NROW), [=](local_int_t i){ \
    if((COLORS)[i] == *color){ \
        const double currentDiagonal = (MATR_DIAG)[i][0]; \
        double sum = (RVALS)[i]; \
        for(int j = 0; j < (NNZ)[i]; j++){ \
          local_int_t curCol = (INDV)[i][j]; \
          sum -= (AMV)[i][j] * (XVALS)[curCol]; \
        } \
        sum += (XVALS)[i]*(MATR_DIAG); \
        (XVALS)[i] = sum/(MATR_DIAG); \
      } \
  }) \
  | then([=](){ *color++; }) \
  | repeat_n(NUM_COLORS)

#define SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \
  then([=](){ *color = 0; }) \
  | SYMGS_SWEEP(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \   
  | repeat_n(FORWARD_AND_BACKWARD)

#define POST_RECURSION_MG(A, r, x, level) \
  PROLONGATION((A), (x), (level)) \
  | then([&](){ \
    ComputeSYMGS_ref((A), (r), (x)); \
  }) \

#define TERMINAL_MG(A, r, x) \
  then([&](){ \
    ZeroVector((x)); \
    ComputeSYMGS_ref((A), (r), (x)); \
  }) \

#define COMPUTE_MG_STAGE1() \
  then([&](){ \
    ZeroVector(z0); \
    ComputeSYMGS_ref(A0, r0, z0); \
  }) \
  | SPMV(A_vals[0], x_vals[0], y_vals[0], A_inds[0], A_nnzs[0], A_nrows[0]) \
  | RESTRICTION(A0, r0, 0) \
  | then([&](){ \
    ZeroVector(z1); \
    ComputeSYMGS_ref(A1, r1, z1); \
  }) \
  | SPMV(A_vals[1], x_vals[1], y_vals[1], A_inds[1], A_nnzs[1], A_nrows[1]) \
  | RESTRICTION(A1, r1, 1) \
  | then([&](){ \
    ZeroVector(z2); \
    ComputeSYMGS_ref(A2, r2, z2); \
  }) \
  | SPMV(A_vals[2], x_vals[2], y_vals[2], A_inds[2], A_nnzs[2], A_nrows[2]) \
  | RESTRICTION(A2, r2, 2)

#define COMPUTE_MG_STAGE2() \
  TERMINAL_MG(A3, r3, z3) \
  | POST_RECURSION_MG(A2, r2, z2, 2) \
  | POST_RECURSION_MG(A1, r1, z1, 1) \
  | POST_RECURSION_MG(A0, r0, z0, 0)

#define MGP0() \
  then([&](){ ZeroVector(z0); }) \
  | SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \
  | SPMV(A_vals[0], x_vals[0], y_vals[0], A_inds[0], A_nnzs[0], A_nrows[0]) \
  | RESTRICTION(A0, r0, 0)

#define MGP1() \
  then([&](){ ZeroVector(z1); }) \
  | SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \
  | SPMV(A_vals[1], x_vals[1], y_vals[1], A_inds[1], A_nnzs[1], A_nrows[1]) \
  | RESTRICTION(A1, r1, 1)

#define MGP2() \
  then([&](){ ZeroVector(z2); }) \
  | SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS) \
  | SPMV(A_vals[2], x_vals[2], y_vals[2], A_inds[2], A_nnzs[2], A_nrows[2]) \
  | RESTRICTION(A2, r2, 2)

#define MGP3() \
  then([&](){ ZeroVector(z3); }) \
  | SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS)

#define MGP4() \
  PROLONGATION(Af, xf, level) \
  | SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS)

#define MGP5() \
  PROLONGATION(Af, xf, level) \
  | SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS)

#define MGP6() \
  PROLONGATION(Af, xf, level) \
  | SYMGS(AMV, XVALS, RVALS, NNZ, INDV, NROW, MATR_DIAG, COLORS)

int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
  const int max_iter, const double tolerance, int &niters, double &normr,  double &normr0,
  double *times, bool doPreconditioning);