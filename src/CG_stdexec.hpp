//COMMIT CREATED TO HIGHLIGHT COMPLEXITY LIMIT OF NVC COMPILER

#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>
#include <numeric>

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"
#include "/opt/nvidia/nsight-systems/2025.3.1/target-linux-x64/nvtx/include/nvtx3/nvtx3.hpp"
#include "ComputeSYMGS_ref.hpp"

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#define NUM_SYMGS_STEPS 7

using stdexec::sender;
using stdexec::then;
using stdexec::schedule;
using stdexec::sync_wait;
using stdexec::bulk;
using stdexec::just;
using stdexec::continues_on;

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
#define SPMV(A, x, y, disable) \
  then([&](){ ExchangeHalo((A), (x)); }) \
  | bulk(stdexec::par_unseq, disable ? 0 : (A).localNumberOfRows, [&](local_int_t i){ \
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
#define SPMV(A, x, y, disable) \
  bulk(stdexec::par_unseq, disable ? 0 : (A).localNumberOfRows, [&](local_int_t i){ \
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

#define RESTRICTION(A, rf, level, disable) \
  bulk(stdexec::par_unseq, disable ? 0 : (A).mgData->rc->localLength, \
    [&](int i){ \
    rcv_ptrs[(level)][i] = rfv_ptrs[(level)][f2c_ptrs[(level)][i]] - Axfv_ptrs[(level)][f2c_ptrs[(level)][i]]; \
  })

#define PROLONGATION(Af, xf, level, disable) \
  bulk(stdexec::par_unseq, disable ? 0 : (Af).mgData->rc->localLength, \
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
  | SPMV((A), (x), *((A).mgData->Axf), false) \
  | RESTRICTION((A), (r), (level), false)

#define POST_RECURSION_MG(A, r, x, level) \
  PROLONGATION((A), (x), (level), false) \
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
  PRE_RECURSION_MG(*Aptrs[0], *rptrs[0], *zptrs[0], 0) \
  | PRE_RECURSION_MG(*Aptrs[1], *rptrs[1], *zptrs[1], 1) \
  | PRE_RECURSION_MG(*Aptrs[2], *rptrs[2], *zptrs[2], 2) \

#define COMPUTE_MG_STAGE2() \
  TERMINAL_MG(*Aptrs[3], *rptrs[3], *zptrs[3]) \
  | POST_RECURSION_MG(*Aptrs[2], *rptrs[2], *zptrs[2], 2) \
  | POST_RECURSION_MG(*Aptrs[1], *rptrs[1], *zptrs[1], 1) \
  | POST_RECURSION_MG(*Aptrs[0], *rptrs[0], *zptrs[0], 0)

#include <iostream>

#define COMPUTE_MG() \
  PROLONGATION(*Aptrs[0], *zptrs[0], 0, prolong_flags[0]) \
  | then([&](){ if(zerovector_flags[0]) ZeroVector(*zptrs[0]); }) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ ComputeSYMGS_ref(*Aptrs[0], *rptrs[0], *zptrs[0]); }) \
  | continues_on(scheduler) \
  | SPMV(*Aptrs[0], *zptrs[0], *((*Aptrs[0]).mgData->Axf), restrict_flags[0]) \
  | RESTRICTION(*Aptrs[0], *rptrs[0], 0, restrict_flags[0]) \
  \
  | PROLONGATION(*Aptrs[1], *zptrs[1], 1, prolong_flags[1]) \
  | then([&](){ if(zerovector_flags[1]) ZeroVector(*zptrs[1]); }) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ ComputeSYMGS_ref(*Aptrs[1], *rptrs[1], *zptrs[1]); }) \
  | continues_on(scheduler) \
  | SPMV(*Aptrs[1], *zptrs[1], *((*Aptrs[1]).mgData->Axf), restrict_flags[1]) \
  | RESTRICTION(*Aptrs[1], *rptrs[1], 1, restrict_flags[1]) \
  \
  | PROLONGATION(*Aptrs[2], *zptrs[2], 2, prolong_flags[2]) \
  | then([&](){ if(zerovector_flags[2]) ZeroVector(*zptrs[2]); }) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ ComputeSYMGS_ref(*Aptrs[2], *rptrs[2], *zptrs[2]); }) \
  | continues_on(scheduler) \
  | SPMV(*Aptrs[2], *zptrs[2], *((*Aptrs[2]).mgData->Axf), restrict_flags[2]) \
  | RESTRICTION(*Aptrs[2], *rptrs[2], 2, restrict_flags[2]) \
  \
  | PROLONGATION(*Aptrs[2], *zptrs[2], 2, prolong_flags[3]) \
  | then([&](){ if(zerovector_flags[3]) ZeroVector(*zptrs[3]); }) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ ComputeSYMGS_ref(*Aptrs[3], *rptrs[3], *zptrs[3]); }) \
  | continues_on(scheduler) \
  | SPMV(*Aptrs[2], *zptrs[2], *((*Aptrs[2]).mgData->Axf), restrict_flags[3]) \
  | RESTRICTION(*Aptrs[2], *rptrs[2], 2, restrict_flags[3]) \
  \
  | PROLONGATION(*Aptrs[4], *zptrs[4], 4, prolong_flags[4]) \
  | then([&](){ if(zerovector_flags[4]) ZeroVector(*zptrs[4]); }) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ ComputeSYMGS_ref(*Aptrs[4], *rptrs[4], *zptrs[4]); }) \
  | continues_on(scheduler) \
  | SPMV(*Aptrs[4], *zptrs[4], *((*Aptrs[4]).mgData->Axf), restrict_flags[4]) \
  | RESTRICTION(*Aptrs[4], *rptrs[4], 4, restrict_flags[4]) \
  \
  | PROLONGATION(*Aptrs[5], *zptrs[5], 5, prolong_flags[5]) \
  | then([&](){ if(zerovector_flags[5]) ZeroVector(*zptrs[5]); }) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ ComputeSYMGS_ref(*Aptrs[5], *rptrs[5], *zptrs[5]); }) \
  | continues_on(scheduler) \
  | SPMV(*Aptrs[5], *zptrs[5], *((*Aptrs[5]).mgData->Axf), restrict_flags[5]) \
  | RESTRICTION(*Aptrs[5], *rptrs[5], 5, restrict_flags[5]) \
  \
  | PROLONGATION(*Aptrs[6], *zptrs[6], 6, prolong_flags[6]) \
  | then([&](){ if(zerovector_flags[6]) ZeroVector(*zptrs[6]); }) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ ComputeSYMGS_ref(*Aptrs[6], *rptrs[6], *zptrs[6]); }) \
  | continues_on(scheduler) \
  | SPMV(*Aptrs[6], *zptrs[6], *((*Aptrs[6]).mgData->Axf), restrict_flags[6]) \
  | RESTRICTION(*Aptrs[6], *rptrs[6], 6, restrict_flags[6])

auto CG_stdexec(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
  const int max_iter, const double tolerance, int & niters, double & normr,  double & normr0,
  double * times, bool doPreconditioning){

  double t_begin = mytimer();  //Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t_dotProd = 0.0, t_WAXPBY = 0.0, t_SPMV = 0.0, t_MG = 0.0 , dummy_time = 0.0;
  double t_zeroVector = 0.0, t_SYMGS = 0.0, t_restrict = 0.0, t_prolong = 0.0, t_tmp = 0.0;
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; //Residual vector
  Vector & z = data.z; //Preconditioned residual vector
  Vector & p = data.p; //Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;
  double local_result;
  double dot_local_copy; //for passing into MPIAllReduce within dot product

  //index used in MG preconditioning loop
  int indPC = 0;

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
  std::vector<local_int_t> prolong_lens(NUM_SYMGS_STEPS, 0);
  std::vector<local_int_t> restrict_lens(NUM_SYMGS_STEPS, 0);
  std::vector<local_int_t> spmv_lens(NUM_SYMGS_STEPS, 0);
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

  //setting lengths of bulk calls in SPMV, prolongation, restriction
  for(int cnt = 0; cnt < NUM_MG_LEVELS - 1; cnt++){
    restrict_lens[cnt] = (*Aptrs[cnt]).mgData->rc->localLength;
    spmv_lens[cnt] = (*Aptrs[cnt]).localNumberOfRows;
  }
  for(int cnt = NUM_MG_LEVELS; cnt < NUM_SYMGS_STEPS; cnt++){
    prolong_lens[cnt] = (*Aptrs[cnt]).mgData->rc->localLength;
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
  //ITERATION FOR FIRST LOOP
  //FIND A MORE ELEGANT WAY OF DOING THIS!
  //NOTE - MUST FIND A MEANS OF MAKING PRECONDITIONING OPTIONAL!

  sender auto my_work = schedule(scheduler)
    | COMPUTE_MG();
  sync_wait(std::move(my_work));

  sender auto rest_of_loop = schedule(scheduler)
    | WAXPBY(1, zVals, 0, zVals, pVals)
    | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz) //rtz = r'*z
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
      niters = 1;
    });
    sync_wait(std::move(rest_of_loop));

  //Start iterations
  //Convergence check accepts an error of no more than 6 significant digits of tolerance
  for(int k = 2; k <= max_iter && normr/normr0 > tolerance; k++){

    sender auto mg_downwards = schedule(scheduler)
    | COMPUTE_MG_STAGE1();
    sync_wait(std::move(mg_downwards));
  
    sender auto mg_upwards = schedule(scheduler)
    | COMPUTE_MG_STAGE2();
    sync_wait(std::move(mg_upwards));

    sender auto rest_of_loop = schedule(scheduler)
    | then([&](){ oldrtz = rtz; })
    | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz) //rtz = r'*z
    | then([&](){ beta = rtz/oldrtz; })
    | WAXPBY(1, zVals, beta, pVals, pVals) //WAXPBY: p = beta*p + z
    | SPMV(A, p, Ap, false) //SPMV: Ap = A*p
    | COMPUTE_DOT_PRODUCT(pVals, ApVals, pAp) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; })
    | WAXPBY(1, xVals, alpha, pVals, xVals) //WAXPBY: x = x + alpha*p
    | WAXPBY(1, rVals, -alpha, ApVals, rVals) //WAXPBY: r = r - alpha*Ap
    | COMPUTE_DOT_PRODUCT(rVals, rVals, normr)
    | then([&](){ 
      normr = sqrt(normr); 
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = k;
    });
    sync_wait(std::move(rest_of_loop));
  }

  sender auto store_times = schedule(scheduler) | then([&](){
    times[1] += t_dotProd; //dot-product time
    times[2] += t_WAXPBY; //WAXPBY time
    times[3] += t_SPMV; //SPMV time
    times[4] += 0.0; //AllReduce time
    times[5] += t_MG; //preconditioner apply time
    times[0] += mytimer() - t_begin;  //Total time
    std::cout << "ADDITIONAL TIME DATA:\n";
    std::cout << "Zero Vector Time : " << t_zeroVector << "\n";
    std::cout << "SYMGS Time : " << t_SYMGS << "\n";
    std::cout << "Restriction Time : " << t_restrict << "\n";
    std::cout << "Prolongation Time : " << t_prolong << "\n";
  });
  sync_wait(std::move(store_times));
  return 0;
}

