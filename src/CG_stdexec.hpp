#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>
#include <numeric>

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"
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

#define NUM_MG_LEVELS 4
#define SINGLE_THREAD 1

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
  bulk(stdexec::par_unseq, (A).localNumberOfRows, [&](local_int_t i){ \
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
  | then([&](){ start_timing("SPMV_stdexec", rangeID); }) \
  | SPMV((A), (x), *((A).mgData->Axf)) \
  | then([&](){ start_timing("Restrict_stdexec", rangeID); }) \
  | RESTRICTION((A), (r), (level)) \
  | then([&](){ end_timing(rangeID); })

#define POST_RECURSION_MG(A, r, x, level) \
  then([&](){ start_timing("Prolong_stdexec", rangeID); }) \
  | PROLONGATION((A), (x), (level)) \
  | then([&](){ end_timing(rangeID); }) \
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

auto CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
  const int max_iter, const double tolerance, int &niters, double &normr,  double &normr0,
  double *times, bool doPreconditioning){

  double t_begin = mytimer();  //start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t_dotProd = 0.0, t_WAXPBY = 0.0, t_SPMV = 0.0, t_MG = 0.0 , dummy_time = 0.0;
  double t_SYMGS = 0.0, t_restrict = 0.0, t_prolong = 0.0, t_tmp = 0.0;
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; //residual vector
  Vector & z = data.z; //preconditioned residual vector
  Vector & p = data.p; //direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;
  double local_result;
  double dot_local_copy; //for passing into MPIAllReduce within dot product

  //variables needed for MG computation
  std::vector<const SparseMatrix*> matrix_ptrs(NUM_MG_LEVELS);
  std::vector<const Vector*> res_ptrs(NUM_MG_LEVELS);
  std::vector<Vector*> zval_ptrs(NUM_MG_LEVELS);
  std::vector<double*> Axfv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> rfv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> rcv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<local_int_t*> f2c_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> xfv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> xcv_ptrs(NUM_MG_LEVELS - 1);
  matrix_ptrs[0] = &A;
  res_ptrs[0] = &r;
  zval_ptrs[0] = &z;
  for(int cnt = 1; cnt < NUM_MG_LEVELS; cnt++){
    matrix_ptrs[cnt] = matrix_ptrs[cnt - 1]->Ac;
    res_ptrs[cnt] = matrix_ptrs[cnt - 1]->mgData->rc;
    zval_ptrs[cnt] = matrix_ptrs[cnt - 1]->mgData->xc;
  }
  for(int cnt = 0; cnt < NUM_MG_LEVELS - 1; cnt++){
    Axfv_ptrs[cnt] = matrix_ptrs[cnt]->mgData->Axf->values;
    rfv_ptrs[cnt] = res_ptrs[cnt]->values;
    rcv_ptrs[cnt] = matrix_ptrs[cnt]->mgData->rc->values;
    f2c_ptrs[cnt] = matrix_ptrs[cnt]->mgData->f2cOperator;
    xfv_ptrs[cnt] = zval_ptrs[cnt]->values;
    xcv_ptrs[cnt] = matrix_ptrs[cnt]->mgData->xc->values;
  }

  //caching dereferenced pointers
  auto& A0 = *matrix_ptrs[0];
  auto& A1 = *matrix_ptrs[1];
  auto& A2 = *matrix_ptrs[2];
  auto& A3 = *matrix_ptrs[3];

  auto& r0 = *res_ptrs[0];
  auto& r1 = *res_ptrs[1];
  auto& r2 = *res_ptrs[2];
  auto& r3 = *res_ptrs[3];

  auto& z0 = *zval_ptrs[0];
  auto& z1 = *zval_ptrs[1];
  auto& z2 = *zval_ptrs[2];
  auto& z3 = *zval_ptrs[3];

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

  nvtxRangeId_t rangeID = 0;

  sender auto pre_loop_work = schedule(scheduler)
  | then([&](){ start_timing("WAXPBY_stdexec", rangeID); })
  | WAXPBY(1, xVals, 0, xVals, pVals)
  | then([&](){ start_timing("SPMV_stdexec", rangeID); })
  | SPMV(A, p, Ap) //SPMV: Ap = A*p
  | then([&](){ start_timing("WAXPBY_stdexec", rangeID); })
  | WAXPBY(1, bVals, -1, ApVals, rVals) //WAXPBY: r = b - Ax (x stored in p)
  | then([&](){ start_timing("DotProd_stdexec", rangeID); })
  | COMPUTE_DOT_PRODUCT(rVals, rVals, normr)
  | then([&](){
    end_timing(rangeID);
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif
    normr0 = normr; //record initial residual for convergence testing
  });
  sync_wait(std::move(pre_loop_work));
  
  int k = 1;
  //ITERATION FOR FIRST LOOP
  sender auto mg_downwards = schedule(scheduler)
    | COMPUTE_MG_STAGE1();
  sync_wait(std::move(mg_downwards));
  
  sender auto mg_upwards = schedule(scheduler)
    | COMPUTE_MG_STAGE2();
  sync_wait(std::move(mg_upwards));

  sender auto rest_of_loop = schedule(scheduler)
    | then([&](){ start_timing("WAXPBY_stdexec", rangeID); })
    | WAXPBY(1, zVals, 0, zVals, pVals)
    | then([&](){ start_timing("DotProd_stdexec", rangeID); })
    | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz) //rtz = r'*z
    | then([&](){ start_timing("SPMV_stdexec", rangeID); })
    | SPMV(A, p, Ap) //SPMV: Ap = A*p
    | then([&](){ start_timing("DotProd_stdexec", rangeID); })
    | COMPUTE_DOT_PRODUCT(pVals, ApVals, pAp) //alpha = p'*Ap
    | then([&](){ 
      end_timing(rangeID);
      alpha = rtz/pAp;
      start_timing("WAXPBY_stdexec", rangeID);
    })
    | WAXPBY(1, xVals, alpha, pVals, xVals) //WAXPBY: x = x + alpha*p
    | then([&](){ start_timing("WAXPBY_stdexec", rangeID); })
    | WAXPBY(1, rVals, -alpha, ApVals, rVals) //WAXPBY: r = r - alpha*Ap
    | then([&](){ start_timing("DotProd_stdexec", rangeID); })
    | COMPUTE_DOT_PRODUCT(rVals, rVals, normr)
    | then([&](){ 
      end_timing(rangeID);
      normr = sqrt(normr);
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (1 % print_freq == 0 || 1 == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = 1;
    });
    sync_wait(std::move(rest_of_loop));

  //start iterations
  //convergence check accepts an error of no more than 6 significant digits of tolerance
  for(int k = 2; k <= max_iter && normr/normr0 > tolerance; k++){

    sender auto mg_downwards = schedule(scheduler)
    | COMPUTE_MG_STAGE1();
    sync_wait(std::move(mg_downwards));
  
    sender auto mg_upwards = schedule(scheduler)
    | COMPUTE_MG_STAGE2();
    sync_wait(std::move(mg_upwards));

    sender auto rest_of_loop = schedule(scheduler)
    | then([&](){ oldrtz = rtz; start_timing("DotProd_stdexec", rangeID); })
    | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz) //rtz = r'*z
    | then([&](){ beta = rtz/oldrtz; start_timing("WAXPBY_stdexec", rangeID); })
    | WAXPBY(1, zVals, beta, pVals, pVals) //WAXPBY: p = beta*p + z
    | then([&](){ start_timing("SPMV_stdexec", rangeID); })
    | SPMV(A, p, Ap) //SPMV: Ap = A*p
    | then([&](){ start_timing("DotProd_stdexec", rangeID); })
    | COMPUTE_DOT_PRODUCT(pVals, ApVals, pAp) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; start_timing("WAXPBY_stdexec", rangeID); })
    | WAXPBY(1, xVals, alpha, pVals, xVals) //WAXPBY: x = x + alpha*p
    | then([&](){ start_timing("WAXPBY_stdexec", rangeID); })
    | WAXPBY(1, rVals, -alpha, ApVals, rVals) //WAXPBY: r = r - alpha*Ap
    | then([&](){ start_timing("DotProd_stdexec", rangeID); })
    | COMPUTE_DOT_PRODUCT(rVals, rVals, normr)
    | then([&](){ 
      end_timing(rangeID);
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
    times[0] += mytimer() - t_begin;  //total time
    std::cout << "ADDITIONAL TIME DATA:\n";
    std::cout << "SYMGS Time : " << t_SYMGS << "\n";
    std::cout << "Restriction Time : " << t_restrict << "\n";
    std::cout << "Prolongation Time : " << t_prolong << "\n";
  });
  sync_wait(std::move(store_times));
  return 0;
}

