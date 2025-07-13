#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>
#include <numeric>
#include <iostream>

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"
#include "../stdexec/include/exec/repeat_n.hpp"
#include "../stdexec/include/exec/variant_sender.hpp"
#include "../stdexec/include/exec/repeat_effect_until.hpp"
#include "/opt/nvidia/nsight-systems/2025.3.1/target-linux-x64/nvtx/include/nvtx3/nvtx3.hpp"
#include "ComputeSYMGS_ref.hpp"

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
using exec::repeat_n;

#define NUM_MG_LEVELS 4
#define SINGLE_THREAD 1

#ifdef NVTX_ON
#define NVTX_RANGE_BEGIN(message) nvtxRangePushA((message));
#define NVTX_RANGE_END nvtxRangePop();
#else
#define NVTX_RANGE_BEGIN(message)
#define NVTX_RANGE_END
#endif

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

#ifndef HPCG_NO_MPI
#define SPMV(A, x, y) \
  then([&](){ ExchangeHalo((A), (x)); }) \
  | bulk(stdexec::par_unseq, (A).localNumberOfRows, [&](local_int_t i){ \
    double sum = 0.0; \
    double *cur_vals = (A).matrixValues[i]; \
    local_int_t *cur_inds = (A).mtxIndL[i]; \
    int cur_nnz = (A).nonzerosInRow[i]; \
    double *xv = (x).values; \
    for(int j = 0; j < cur_nnz; j++) \
      sum += cur_vals[j]*xv[cur_inds[j]]; \
    (y).values[i] = sum; \
  })
#else
#define SPMV(A, x, y) \
  bulk(stdexec::par_unseq, (A).localNumberOfRows, [&](local_int_t i){ \
    double sum = 0.0; \
    double *cur_vals = (A).matrixValues[i]; \
    local_int_t *cur_inds = (A).mtxIndL[i]; \
    int cur_nnz = (A).nonzerosInRow[i]; \
    double *xv = (x).values; \
    for(int j = 0; j < cur_nnz; j++) \
      sum += cur_vals[j]*xv[cur_inds[j]]; \
    (y).values[i] = sum; \
  })
#endif

#define RESTRICTION(A, rf, level) \
  bulk(stdexec::par_unseq, (A).mgData->rc->localLength, [&](int i){ \
    rcv_ptrs[(level)][i] = rfv_ptrs[(level)][f2c_ptrs[(level)][i]] - Axfv_ptrs[(level)][f2c_ptrs[(level)][i]]; \
  })

#define PROLONGATION(Af, xf, level) \
  bulk(stdexec::par_unseq, (Af).mgData->rc->localLength, [&](int i){ \
    xfv_ptrs[(level)][f2c_ptrs[(level)][i]] += xcv_ptrs[(level)][i]; \
  })

#define SYMGS(A, r, x) \
  continues_on(scheduler_single_thread) \
  | then([&](){ ComputeSYMGS_ref((A), (r), (x)); }) \
  | continues_on(scheduler)

//NOTE - OMITTED MPI HALOEXCHANGE IN SYMGS
#define PRE_RECURSION_MG(A, r, x, level) \
  then([&](){ ZeroVector((x)); }) \
  | SYMGS((A), (r), (x)) \
  | SPMV((A), (x), *((A).mgData->Axf)) \
  | RESTRICTION((A), (r), (level))

#define POST_RECURSION_MG(A, r, x, level) \
  PROLONGATION((A), (x), (level)) \
  | SYMGS((A), (r), (x))

#define TERMINAL_MG(A, r, x) \
  then([&](){ ZeroVector((x)); }) \
  | SYMGS((A), (r), (x))

auto CG_stdexec(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
  const int max_iter, const double tolerance, int & niters, double & normr,  double & normr0,
  double * times, bool doPreconditioning){

  double t_begin = mytimer();  //Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t_dotProd = 0.0, t_WAXPBY = 0.0, t_SPMV = 0.0, t_MG = 0.0 ;
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; //Residual vector
  Vector & z = data.z; //Preconditioned residual vector
  Vector & p = data.p; //Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;
  double local_result;

  //declaring all the variables needed for MG computation
  std::vector<const SparseMatrix*> Aptrs(NUM_MG_LEVELS);
  std::vector<const Vector*> rptrs(NUM_MG_LEVELS);
  std::vector<Vector*> zptrs(NUM_MG_LEVELS);
  std::vector<double*> Axfv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> rfv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> rcv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<local_int_t*> f2c_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> xfv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> xcv_ptrs(NUM_MG_LEVELS - 1);

  //setting the values of A, r and z 
  Aptrs[0] = &A;
  rptrs[0] = &r;
  zptrs[0] = &z;
  for(int cnt = 1; cnt < NUM_MG_LEVELS; cnt++){
    Aptrs[cnt] = Aptrs[cnt - 1]->Ac;
    rptrs[cnt] = Aptrs[cnt - 1]->mgData->rc;
    zptrs[cnt] = Aptrs[cnt - 1]->mgData->xc;
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
  | SPMV(A, p, Ap) //SPMV: Ap = A*p
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

  sender auto mg_stage1 = schedule(scheduler) | PRE_RECURSION_MG(*Aptrs[0], *rptrs[0], *zptrs[0], 0);
  sender auto mg_stage2 = schedule(scheduler) | PRE_RECURSION_MG(*Aptrs[1], *rptrs[1], *zptrs[1], 1);
  sender auto mg_stage3 = schedule(scheduler) | PRE_RECURSION_MG(*Aptrs[2], *rptrs[2], *zptrs[2], 2);
  sender auto mg_stage4 = schedule(scheduler) | TERMINAL_MG(*Aptrs[3], *rptrs[3], *zptrs[3]);
  sender auto mg_stage5 = schedule(scheduler) | POST_RECURSION_MG(*Aptrs[2], *rptrs[2], *zptrs[2], 2);
  sender auto mg_stage6 = schedule(scheduler) | POST_RECURSION_MG(*Aptrs[1], *rptrs[1], *zptrs[1], 1);
  sender auto mg_stage7 = schedule(scheduler) | POST_RECURSION_MG(*Aptrs[0], *rptrs[0], *zptrs[0], 0);

  exec::variant_sender<decltype(mg_stage1),
                        decltype(mg_stage2),
                        decltype(mg_stage3),
                        decltype(mg_stage4),
                        decltype(mg_stage5),
                        decltype(mg_stage6),
                        decltype(mg_stage7)> switch_mg = mg_stage1;

  //index used in MG preconditioning loop
  int indPC = 1;

  sender auto compute_mg = let_value(schedule(scheduler), [&](){
    switch(indPC){
      case 1:
        switch_mg.emplace<0>(mg_stage1);
        return switch_mg;
      case 2:
        switch_mg.emplace<1>(mg_stage2);
        return switch_mg;
      case 3:
        switch_mg.emplace<2>(mg_stage3);
        return switch_mg;
      case 4:
        switch_mg.emplace<3>(mg_stage4);
        return switch_mg;
      case 5:
        switch_mg.emplace<4>(mg_stage5);
        return switch_mg;
      case 6:
        switch_mg.emplace<5>(mg_stage6);
        return switch_mg;
      case 7:
        switch_mg.emplace<6>(mg_stage7);
        return switch_mg;
    }
  })
  | repeat_n(7);

  sender auto sndr_first = compute_mg
  | WAXPBY(1, zVals, 0, zVals, pVals)
  | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz); //rtz = r'*z

  sender auto sndr_second = compute_mg
  | then([&](){ oldrtz = rtz; })
  | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz) //rtz = r'*z
  | then([&](){ beta = rtz/oldrtz; })
  | WAXPBY(1, zVals, beta, pVals, pVals); //WAXPBY: p = beta*p + z

  exec::variant_sender<decltype(sndr_first), decltype(sndr_second)> switch_sndr = sndr_first;

  sender auto loop_work = let_value(schedule(scheduler), [&](){
    if(k == 2){
      switch_sndr.emplace<1>(sndr_second);
      return switch_sndr;
    }else{
      return switch_sndr;
    }
  })
  | SPMV(A, p, Ap) //SPMV: Ap = A*p
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
    indPC = 1;
    std::cout << k << "\n";
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
  });
  sync_wait(std::move(store_times));
  return 0;
}