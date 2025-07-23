#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include "ComputeSYMGS_ref.hpp"
#include <cassert>
#include "NVTX_timing.hpp"

//timing stuff
#include <chrono>
#include <omp.h>
#include <iostream>
#define tstart t_start = std::chrono::high_resolution_clock::now()
#define tend t_end = std::chrono::high_resolution_clock::now()
#define NUM_THR 4
#define NUM_STEPS 8
#define tagg1 tend; step_times[0][tid] += std::chrono::duration<double>(t_end - t_start).count()
#define tagg2 tend; step_times[1][tid] += std::chrono::duration<double>(t_end - t_start).count()
#define tagg3 tend; step_times[2][tid] += std::chrono::duration<double>(t_end - t_start).count()
#define tagg4 tend; step_times[3][tid] += std::chrono::duration<double>(t_end - t_start).count()
#define tagg5 tend; step_times[4][tid] += std::chrono::duration<double>(t_end - t_start).count()
#define tagg6 tend; step_times[5][tid] += std::chrono::duration<double>(t_end - t_start).count()
#define tagg7 tend; step_times[6][tid] += std::chrono::duration<double>(t_end - t_start).count()
#define tagg8 tend; step_times[7][tid] += std::chrono::duration<double>(t_end - t_start).count()

int ComputeSYMGS_opt(const SparseMatrix &A, const Vector &r, Vector &x){
  nvtxRangeId_t rangeID = 0;
  start_timing("SYMGS_opt", rangeID);
  assert(x.localLength == A.localNumberOfColumns);
#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif
  
  //timing stuff
  std::vector<std::vector<double>> step_times;
  step_times.resize(NUM_STEPS, std::vector<double>(NUM_THR, 0.0));

  const double * const rv = r.values;
  double * const xv = x.values;

  for(int color = 0; color < NUM_COLORS; color++){
    local_int_t startInd = A.startInds[color];
    local_int_t endInd = A.endInds[color];

    #pragma omp parallel for
    for(local_int_t ind = startInd; ind <= endInd; ind++){

      //timing stuff
      std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
      std::chrono::time_point<std::chrono::high_resolution_clock> t_end;
      int tid = omp_get_thread_num();

      tstart; RowDataFlat *rowPtr = &A.rowStructs[ind]; tagg1;
      tstart; double *vals = rowPtr->values; tagg2;
      tstart; local_int_t *cols = rowPtr->cols; tagg3;
      tstart; int nnz = rowPtr->numNonzeros; tagg4;
      tstart; double diagVal = rowPtr->diagVal; tagg5;
      tstart; int i = rowPtr->rowIndex; tagg6;
      double sum = rv[i]; //RHS value

      tstart; 
      for (int j = 0; j < nnz; j++){
        local_int_t curCol = cols[j];
        sum -= vals[j]*xv[curCol];
      }
      tagg7;

      tstart; 
      sum += xv[i]*diagVal; //remove diagonal contribution from previous loop
      xv[i] = sum/diagVal;
      tagg8;
    }
  }

  //back sweep
  for(int color = NUM_COLORS - 1; color >= 0; color--){
    local_int_t startInd = A.startInds[color];
    local_int_t endInd = A.endInds[color];

    #pragma omp parallel for
    for(local_int_t ind = startInd; ind <= endInd; ind++){
      RowDataFlat *rowPtr = &A.rowStructs[ind];
      double *vals = rowPtr->values;
      local_int_t *cols = rowPtr->cols;
      int nnz = rowPtr->numNonzeros;
      double diagVal = rowPtr->diagVal;
      int i = rowPtr->rowIndex;
      double sum = rv[i]; //RHS value

      for(int j = 0; j < nnz; j++){
        local_int_t curCol = cols[j];
        sum -= vals[j]*xv[curCol];
      }
      sum += xv[i]*diagVal; //remove diagonal contribution from previous loop
      xv[i] = sum/diagVal;
    }
  }

  //more timing stuff
  for(int stepCnt = 0; stepCnt < NUM_STEPS; stepCnt++){
    double time_sum = 0.0;
    for(int cnt = 0; cnt < NUM_THR; cnt++)
      time_sum += step_times[stepCnt][cnt];
    std::cout << "Time spent at step " << stepCnt << " = " << time_sum << ".\n";
  }
  std::cout << "\n";

  end_timing(rangeID);
  return 0;
}

