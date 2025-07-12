#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>
#include <numeric>
#include <iostream>
#include <fstream>

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"
#include "../stdexec/include/exec/repeat_n.hpp"
#include "../stdexec/include/exec/variant_sender.hpp"
#include "../stdexec/include/exec/repeat_effect_until.hpp"
#include "/opt/nvidia/nsight-systems/2025.3.1/target-linux-x64/nvtx/include/nvtx3/nvtx3.hpp"
#include "CG_stdexec.hpp"
#include "ComputeSYMGS_ref.hpp"
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
using stdexec::let_value;
using exec::repeat_effect_until;

#define NUM_MG_LEVELS 4
#define NUM_SYMGS_STEPS 7
#define SINGLE_THREAD 1

#ifdef NVTX_ON
#define NVTX_RANGE_BEGIN(message) nvtxRangePushA((message));
#define NVTX_RANGE_END nvtxRangePop();
#else
#define NVTX_RANGE_BEGIN(message)
#define NVTX_RANGE_END
#endif

//index used in MG preconditioning loop
int indPC = 0;
const SparseMatrix* A_spmv; 
Vector* x_spmv; 
Vector* y_spmv;
bool disable_spmv;

#ifndef HPCG_NO_MPI
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([&](){ \
    local_result = 0.0; \
    local_result = std::transform_reduce(std::execution::par, (VEC1VALS), (VEC1VALS) + nrow, (VEC2VALS), 0.0); \
    MPI_Allreduce(&local_result, &(RESULT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); \
  })
#else
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([&](){ \
    (RESULT) = std::transform_reduce(std::execution::par, (VEC1VALS), (VEC1VALS) + nrow, (VEC2VALS), 0.0); \
  })
#endif

#define WAXPBY(ALPHA, XVALS, BETA, YVALS, WVALS) \
  bulk(stdexec::par_unseq, nrow, [&](local_int_t i){ (WVALS)[i] = (ALPHA)*(XVALS)[i] + (BETA)*(YVALS)[i]; })

void spmv_kernel(local_int_t i){
  if(disable_spmv){ return; } 
  if(i >= A_spmv->localNumberOfRows){ return; }
  double sum = 0.0;
  double *cur_vals = A_spmv->matrixValues[i];
  local_int_t *cur_inds = A_spmv->mtxIndL[i];
  int cur_nnz = A_spmv->nonzerosInRow[i];
  double *xv = x_spmv->values;
  for(int j = 0; j < cur_nnz; j++)
    sum += cur_vals[j]*xv[cur_inds[j]];
  y_spmv->values[i] = sum;
}

#ifndef HPCG_NO_MPI
#define SPMV(A, x, y, disable) \
  then([&](){ ExchangeHalo((A), (x)); }) \
  | bulk(stdexec::par_unseq, disable ? 0 : (A).localNumberOfRows, spmv_kernel)
#else
#define SPMV(A, x, y, disable) \
  then([&](){ t_tmp = mytimer(); A_spmv = &(A); x_spmv = &(x); if(!disable){ y_spmv = &(y); } disable_spmv = (disable); }) \
  | bulk(stdexec::par_unseq, disable ? 0 : (A).localNumberOfRows, spmv_kernel) \
  | then([&](){ t_SPMV += mytimer() - t_tmp; })
#endif

#define RESTRICTION(A, rf, level, disable) \
  then([&](){ t_tmp = mytimer(); }) \
  | bulk(stdexec::par_unseq, disable ? 0 : (A).mgData->rc->localLength, \
    [&](int i){ \
    if(disable){ return; } \
    if(i >= (A).mgData->rc->localLength){ return; } \
    rcv_ptrs[(level)][i] = rfv_ptrs[(level)][f2c_ptrs[(level)][i]] - Axfv_ptrs[(level)][f2c_ptrs[(level)][i]]; \
  }) \
  | then([&](){ t_restrict += mytimer() - t_tmp; })

#define PROLONGATION(Af, xf, level, disable) \
  then([&](){ t_tmp = mytimer(); }) \
  | bulk(stdexec::par_unseq, disable ? 0 : (Af).mgData->rc->localLength, \
    [&](int i){ \
    if(disable){ return; } \
    if(i >= (A).mgData->rc->localLength){ return; } \
    xfv_ptrs[(level)][f2c_ptrs[(level)][i]] += xcv_ptrs[(level)][i]; \
  }) \
  | then([&](){ t_prolong += mytimer() - t_tmp; })

#define COMPUTE_MG() \
  PROLONGATION(*Aptrs[indPC], *zptrs[indPC], indPC, prolong_flags[indPC]) \
  | then([&](){ if(zerovector_flags[indPC]) ZeroVector(*zptrs[indPC]); }) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ ComputeSYMGS_ref(*Aptrs[indPC], *rptrs[indPC], *zptrs[indPC]); }) \
  | continues_on(scheduler) \
  | SPMV(*Aptrs[indPC], *zptrs[indPC], *((*Aptrs[indPC]).mgData->Axf), restrict_flags[indPC]) \
  | RESTRICTION(*Aptrs[indPC], *rptrs[indPC], indPC, restrict_flags[indPC]) \
  | then([&](){ indPC++; }) \
  | exec::repeat_n(NUM_SYMGS_STEPS)

int CG_stdexec(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
  const int max_iter, const double tolerance, int & niters, double & normr,  double & normr0,
  double * times, bool doPreconditioning){

  double t_begin = mytimer();  //Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t_dotProd = 0.0, t_WAXPBY = 0.0, t_SPMV = 0.0, t_MG = 0.0, t_tmp = 0.0;
  double t_restrict = 0.0, t_prolong = 0.0;
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; //Residual vector
  Vector & z = data.z; //Preconditioned residual vector
  Vector & p = data.p; //Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;
  double local_result;

  //declaring all the variables needed for MG computation
  std::vector<const SparseMatrix*> Aptrs(NUM_SYMGS_STEPS);
  std::vector<const Vector*> rptrs(NUM_SYMGS_STEPS);
  std::vector<Vector*> zptrs(NUM_SYMGS_STEPS);
  std::vector<double*> Axfv_ptrs(NUM_SYMGS_STEPS);
  std::vector<double*> rfv_ptrs(NUM_SYMGS_STEPS);
  std::vector<double*> rcv_ptrs(NUM_SYMGS_STEPS);
  std::vector<local_int_t*> f2c_ptrs(NUM_SYMGS_STEPS);
  std::vector<double*> xfv_ptrs(NUM_SYMGS_STEPS);
  std::vector<double*> xcv_ptrs(NUM_SYMGS_STEPS);
  std::vector<bool> zerovector_flags(NUM_SYMGS_STEPS, false);
  std::vector<bool> restrict_flags(NUM_SYMGS_STEPS, true);
  std::vector<bool> prolong_flags(NUM_SYMGS_STEPS, true);

  //setting the values of A, r and z 
  Aptrs[0] = &A;
  rptrs[0] = &r;
  zptrs[0] = &z;
  for(int cnt = 1; cnt < NUM_MG_LEVELS; cnt++){
    Aptrs[cnt] = Aptrs[cnt - 1]->Ac;
    rptrs[cnt] = Aptrs[cnt - 1]->mgData->rc;
    zptrs[cnt] = Aptrs[cnt - 1]->mgData->xc;
  }
  for(int cnt = NUM_MG_LEVELS; cnt < NUM_SYMGS_STEPS; cnt++){
    Aptrs[cnt] = Aptrs[NUM_SYMGS_STEPS - (cnt+1)];
    rptrs[cnt] = rptrs[NUM_SYMGS_STEPS - (cnt+1)];
    zptrs[cnt] = zptrs[NUM_SYMGS_STEPS - (cnt+1)];
  }

  //setting values for variables used in prolongation and restriction
  for(int cnt = 0; cnt < NUM_MG_LEVELS - 1; cnt++){
    Axfv_ptrs[cnt] = Aptrs[cnt]->mgData->Axf->values;
    rfv_ptrs[cnt] = rptrs[cnt]->values;
    rcv_ptrs[cnt] = Aptrs[cnt]->mgData->rc->values;
    f2c_ptrs[cnt] = Aptrs[cnt]->mgData->f2cOperator;
    xfv_ptrs[cnt] = zptrs[cnt]->values;
    xcv_ptrs[cnt] = Aptrs[cnt]->mgData->xc->values;
  }
  Axfv_ptrs[NUM_MG_LEVELS - 1] = Axfv_ptrs[NUM_MG_LEVELS - 2];
  rfv_ptrs[NUM_MG_LEVELS - 1] = rfv_ptrs[NUM_MG_LEVELS - 2];
  rcv_ptrs[NUM_MG_LEVELS - 1] = rcv_ptrs[NUM_MG_LEVELS - 2];
  f2c_ptrs[NUM_MG_LEVELS - 1] = f2c_ptrs[NUM_MG_LEVELS - 2];
  xfv_ptrs[NUM_MG_LEVELS - 1] = xfv_ptrs[NUM_MG_LEVELS - 2];
  xcv_ptrs[NUM_MG_LEVELS - 1] = xcv_ptrs[NUM_MG_LEVELS - 2];
  for(int cnt = NUM_MG_LEVELS; cnt < NUM_SYMGS_STEPS; cnt++){
    Axfv_ptrs[cnt] = Axfv_ptrs[NUM_SYMGS_STEPS - (cnt+1)];
    rfv_ptrs[cnt] = rfv_ptrs[NUM_SYMGS_STEPS - (cnt+1)];
    rcv_ptrs[cnt] = rcv_ptrs[NUM_SYMGS_STEPS - (cnt+1)];
    f2c_ptrs[cnt] = f2c_ptrs[NUM_SYMGS_STEPS - (cnt+1)];
    xfv_ptrs[cnt] = xfv_ptrs[NUM_SYMGS_STEPS - (cnt+1)];
    xcv_ptrs[cnt] = xcv_ptrs[NUM_SYMGS_STEPS - (cnt+1)];
  }

  //set zerovector flags
  for(int cnt = 0; cnt < NUM_MG_LEVELS; cnt++){
    zerovector_flags[cnt] = true;
  }

  //set restriction flags
  for(int cnt = 0; cnt < NUM_MG_LEVELS - 1; cnt++){
    restrict_flags[cnt] = false;
  }

  //set prolongation flags
  for(int cnt = NUM_MG_LEVELS; cnt < NUM_SYMGS_STEPS; cnt++){
    prolong_flags[cnt] = false;
  }

  //used in dot product and WAXPBY calculations
  double *rVals = r.values;
  double *zVals = z.values;
  double *pVals = p.values;
  double *xVals = x.values;
  double *bVals = b.values;
  double *ApVals = Ap.values;

  //scheduler for CPU execution
  unsigned int num_threads = std::thread::hardware_concurrency();
  if(num_threads == 0){
    std::cerr << "ERROR: CANNOT DETERMINE THREAD POOL SIZE.\n";
    std::exit(EXIT_FAILURE);
  }
  else{
    std::cout << "THREAD POOL SIZE IS " << num_threads << ".\n";
  }
  exec::static_thread_pool pool(num_threads);
  auto scheduler = pool.get_scheduler();

  //scheduler for SYMGS execution
  exec::static_thread_pool pool_single_thread(SINGLE_THREAD);
  auto scheduler_single_thread = pool_single_thread.get_scheduler();

  if (!doPreconditioning && A.geom->rank == 0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
#endif

  sender auto pre_loop_work = schedule(scheduler)
  | WAXPBY(1, xVals, 0, xVals, pVals)
  | SPMV(A, p, Ap, false) //SPMV: Ap = A*p
  | WAXPBY(1, bVals, -1, ApVals, rVals) //WAXPBY: r = b - Ax (x stored in p)
  | COMPUTE_DOT_PRODUCT(rVals, rVals, normr)
  | then([&](){
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif
    normr0 = normr; //Record initial residual for convergence testing
  });
  sync_wait(std::move(pre_loop_work));
  
  int k = 1;

  stdexec::sender auto sndr_first = stdexec::schedule(scheduler) 
  | COMPUTE_MG()
  | WAXPBY(1, zVals, 0, zVals, pVals)
  | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz); //rtz = r'*z

  stdexec::sender auto sndr_second = stdexec::schedule(scheduler) 
  | COMPUTE_MG()
  | then([&](){ oldrtz = rtz; })
  | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz) //rtz = r'*z
  | then([&](){ beta = rtz/oldrtz; })
  | WAXPBY(1, zVals, beta, pVals, pVals); //WAXPBY: p = beta*p + z

  using loop_one_t = decltype(sndr_first);
  using loop_two_t = decltype(sndr_second);
  exec::variant_sender<loop_one_t, loop_two_t> switch_sndr = sndr_first;

  sender auto loop_work = let_value(stdexec::schedule(scheduler), [&](){
    if(k == 2){
      switch_sndr.emplace<1>(sndr_second);
      return switch_sndr;
    }else{
      return switch_sndr;
    }
  })
  | SPMV(A, p, Ap, false) //SPMV: Ap = A*p
  | COMPUTE_DOT_PRODUCT(pVals, ApVals, pAp) //alpha = p'*Ap
  | then([&](){ alpha = rtz/pAp; })
  | WAXPBY(1, xVals, alpha, pVals, xVals) //WAXPBY: x = x + alpha*p
  | WAXPBY(1, rVals, -alpha, ApVals, rVals) //WAXPBY: r = r - alpha*Ap
  | COMPUTE_DOT_PRODUCT(rVals, rVals, normr)
  | then([&](){ 
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0 && (1 % print_freq == 0 || 1 == max_iter))
      HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
    niters = k;
    k++;
    indPC = 0;
    return !(k <= max_iter && normr/normr0 > tolerance);
  })
  | repeat_effect_until();
  
  sync_wait(std::move(loop_work));

  sender auto store_times = schedule(scheduler) | then([&](){
    times[1] += t_dotProd; //dot-product time
    times[2] += t_WAXPBY; //WAXPBY time
    times[3] += t_SPMV; //SPMV time
    times[4] += 0.0; //AllReduce time
    times[5] += t_MG; //preconditioner apply time
    times[0] += mytimer() - t_begin;  //Total time

    std::cout << "t_restrict: " << t_restrict << "\n";
    std::cout << "t_prolong: " << t_prolong << "\n";
  });
  sync_wait(std::move(store_times));
  return 0;
}