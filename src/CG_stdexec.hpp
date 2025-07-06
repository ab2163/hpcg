#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "../stdexec/include/exec/static_thread_pool.hpp"
#include "/opt/nvidia/nsight-systems/2025.3.1/target-linux-x64/nvtx/include/nvtx3/nvtx3.hpp"

using stdexec::sender;
using stdexec::then;
using stdexec::schedule;
using stdexec::sync_wait;
using stdexec::bulk;
using stdexec::just;
using stdexec::continues_on;

#ifdef TIMING_ON
#define TIMING_WRAPPER(func, timeVar) \
  (std::invoke([&](){ \
    double startTime = mytimer(); \
    sync_wait(schedule(TIMING_SCHEDULER) | (func)); \
    (timeVar) += mytimer() - startTime; \
    return then([](){}); \
  }))
#else
#define TIMING_WRAPPER(func, timeVar) (func)
#endif

#define TW TIMING_WRAPPER
#define TIMING_SCHEDULER scheduler
#define NUM_MG_LEVELS 4
#define SINGLE_THREAD 1

#ifdef TIMING_ON
#define NVTX_RANGE_BEGIN(message) nvtxRangePushA((message));
#define NVTX_RANGE_END nvtxRangePop();
#else
#define NVTX_RANGE_BEGIN(message)
#define NVTX_RANGE_END
#endif

#ifndef HPCG_NO_MPI
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([&](){ dot_local_result = 0.0; }) \
  | bulk(stdexec::par, nrow, [&](local_int_t i){ \
    dot_local_result.fetch_add((VEC1VALS)[i]*(VEC2VALS)[i], std::memory_order_relaxed); }) \
  | then([&](){ \
    dot_local_copy = dot_local_result.load(); \
    MPI_Allreduce(&dot_local_copy, &(RESULT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);})
#else
#define COMPUTE_DOT_PRODUCT(VEC1VALS, VEC2VALS, RESULT) \
  then([&](){ dot_local_result = 0.0; }) \
  | bulk(stdexec::par, nrow, [&](local_int_t i){ \
    dot_local_result.fetch_add((VEC1VALS)[i]*(VEC2VALS)[i], std::memory_order_relaxed); }) \
  | then([&](){ (RESULT) = dot_local_result.load(); })
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

#define SYMGS(A, r, x) \
  NVTX_RANGE_BEGIN("ComputeSYMGS_stdexec") \
  nrow_SYMGS = (A).localNumberOfRows; \
  matrixDiagonal = (A).matrixDiagonal; \
  rv = (r).values; \
  xv = (x).values; \
  for(local_int_t i = 0; i < nrow_SYMGS; i++){ \
    const double * const currentValues = (A).matrixValues[i]; \
    const local_int_t * const currentColIndices = (A).mtxIndL[i]; \
    const int currentNumberOfNonzeros = (A).nonzerosInRow[i]; \
    const double  currentDiagonal = matrixDiagonal[i][0]; \
    double sum = rv[i]; \
    NVTX_RANGE_BEGIN("ComputeSYMGS_stdexec_forward_for") \
    for(int j = 0; j < currentNumberOfNonzeros; j++){ \
      local_int_t curCol = currentColIndices[j]; \
      sum -= currentValues[j] * xv[curCol]; \
    } \
    NVTX_RANGE_END \
    sum += xv[i]*currentDiagonal; \
    xv[i] = sum/currentDiagonal; \
  } \
  for(local_int_t i = nrow_SYMGS - 1; i >= 0; i--){ \
    const double * const currentValues = (A).matrixValues[i]; \
    const local_int_t * const currentColIndices = (A).mtxIndL[i]; \
    const int currentNumberOfNonzeros = (A).nonzerosInRow[i]; \
    const double  currentDiagonal = matrixDiagonal[i][0]; \
    double sum = rv[i]; \
    NVTX_RANGE_BEGIN("ComputeSYMGS_stdexec_backwards_for") \
    for(int j = 0; j < currentNumberOfNonzeros; j++){ \
      local_int_t curCol = currentColIndices[j]; \
      sum -= currentValues[j]*xv[curCol]; \
    } \
    NVTX_RANGE_END \
    sum += xv[i]*currentDiagonal; \
    xv[i] = sum/currentDiagonal; \
  } \
  NVTX_RANGE_END

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
    t_tmp = mytimer(); \
    ZeroVector((x)); \
    t_zeroVector += mytimer() - t_tmp; \
    t_tmp = mytimer(); \
    SYMGS((A), (r), (x)) \
    t_SYMGS += mytimer() - t_tmp; \
  }) \
  | continues_on(scheduler) \
  | TW(SPMV((A), (x), *((A).mgData->Axf)), t_SPMV) \
  | TW(RESTRICTION((A), (r), (level)), t_restrict)

#define POST_RECURSION_MG(A, r, x, level) \
  TW(PROLONGATION((A), (x), (level)), t_prolong) \
  | continues_on(scheduler_single_thread) \
  | then([&](){ \
    t_tmp = mytimer(); \
    SYMGS((A), (r), (x)) \
    t_SYMGS += mytimer() - t_tmp; \
  }) \
  | continues_on(scheduler)

#define TERMINAL_MG(A, r, x) \
  continues_on(scheduler_single_thread) \
  | then([&](){ \
    t_tmp = mytimer(); \
    ZeroVector((x)); \
    t_zeroVector += mytimer() - t_tmp; \
    t_tmp = mytimer(); \
    SYMGS((A), (r), (x)) \
    t_SYMGS += mytimer() - t_tmp; \
  }) \
  | continues_on(scheduler)
  
#define COMPUTE_MG() \
  PRE_RECURSION_MG(*matrix_ptrs[0], *res_ptrs[0], *zval_ptrs[0], 0) \
  | PRE_RECURSION_MG(*matrix_ptrs[1], *res_ptrs[1], *zval_ptrs[1], 1) \
  | PRE_RECURSION_MG(*matrix_ptrs[2], *res_ptrs[2], *zval_ptrs[2], 2) \
  | TERMINAL_MG(*matrix_ptrs[3], *res_ptrs[3], *zval_ptrs[3]) \
  | POST_RECURSION_MG(*matrix_ptrs[2], *res_ptrs[2], *zval_ptrs[2], 2) \
  | POST_RECURSION_MG(*matrix_ptrs[1], *res_ptrs[1], *zval_ptrs[1], 1) \
  | POST_RECURSION_MG(*matrix_ptrs[0], *res_ptrs[0], *zval_ptrs[0], 0)

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
  std::atomic<double> dot_local_result(0.0); //for summing within dot product
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

  //used in SYMGS - declare here to avoid redeclarations
  local_int_t nrow_SYMGS;
  double **matrixDiagonal;
  double *rv;
  double *xv;

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
  | TW(WAXPBY(1, xVals, 0, xVals, pVals), t_WAXPBY)
  | TW(SPMV(A, p, Ap), t_SPMV) //SPMV: Ap = A*p
  | TW(WAXPBY(1, bVals, -1, ApVals, rVals), t_WAXPBY) //WAXPBY: r = b - Ax (x stored in p)
  | TW(COMPUTE_DOT_PRODUCT(rVals, rVals, normr), t_dotProd)
  | TW(then([&](){
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif
    normr0 = normr; //Record initial residual for convergence testing
  }), dummy_time);
  sync_wait(std::move(pre_loop_work));
  
  int k = 1;
  //ITERATION FOR FIRST LOOP
  //FIND A MORE ELEGANT WAY OF DOING THIS!
  sender auto first_loop = schedule(scheduler)
    //NOTE - MUST FIND A MEANS OF MAKING PRECONDITIONING OPTIONAL!
    | TW(COMPUTE_MG(), t_MG)
    | TW(WAXPBY(1, zVals, 0, zVals, pVals), t_WAXPBY)
    | TW(COMPUTE_DOT_PRODUCT(rVals, zVals, rtz), t_dotProd) //rtz = r'*z
    | TW(SPMV(A, p, Ap), t_SPMV) //SPMV: Ap = A*p
    | TW(COMPUTE_DOT_PRODUCT(pVals, ApVals, pAp), t_dotProd) //alpha = p'*Ap
    | TW(then([&](){ alpha = rtz/pAp; }), dummy_time)
    | TW(WAXPBY(1, xVals, alpha, pVals, xVals), t_WAXPBY) //WAXPBY: x = x + alpha*p
    | TW(WAXPBY(1, rVals, -alpha, ApVals, rVals), t_WAXPBY) //WAXPBY: r = r - alpha*Ap
    | TW(COMPUTE_DOT_PRODUCT(rVals, rVals, normr), t_dotProd)
    | TW(then([&](){ 
      normr = sqrt(normr);
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (1 % print_freq == 0 || 1 == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = 1;
    }), dummy_time);
    sync_wait(std::move(first_loop));

  //Start iterations
  //Convergence check accepts an error of no more than 6 significant digits of tolerance
  for(int k = 2; k <= max_iter && normr/normr0 > tolerance; k++){
    sender auto subsequent_loop = schedule(scheduler)
    | TW(COMPUTE_MG(), t_MG)
    | TW(then([&](){ oldrtz = rtz; }), dummy_time)
    | TW(COMPUTE_DOT_PRODUCT(rVals, zVals, rtz), t_dotProd) //rtz = r'*z
    | TW(then([&](){ beta = rtz/oldrtz; }), dummy_time)
    | TW(WAXPBY(1, zVals, beta, pVals, pVals), t_WAXPBY) //WAXPBY: p = beta*p + z
    | TW(SPMV(A, p, Ap), t_SPMV) //SPMV: Ap = A*p
    | TW(COMPUTE_DOT_PRODUCT(pVals, ApVals, pAp), t_dotProd) //alpha = p'*Ap
    | TW(then([&](){ alpha = rtz/pAp; }), dummy_time)
    | TW(WAXPBY(1, xVals, alpha, pVals, xVals), t_WAXPBY) //WAXPBY: x = x + alpha*p
    | TW(WAXPBY(1, rVals, -alpha, ApVals, rVals), t_WAXPBY) //WAXPBY: r = r - alpha*Ap
    | TW(COMPUTE_DOT_PRODUCT(rVals, rVals, normr), t_dotProd)
    | TW(then([&](){ 
      normr = sqrt(normr); 
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = k;
    }), dummy_time);
    sync_wait(std::move(subsequent_loop));
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

