#include <fstream>
#include <cmath>

#include "CGData.hpp"
#include "hpcg.hpp"
#include "ComputeMG_stdexec.hpp"
#include "ComputeDotProduct_stdexec.hpp"
#include "ComputeWAXPBY_stdexec.hpp"

//Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

using stdexec::sender;
using stdexec::then;
using stdexec::schedule;
using stdexec::sync_wait;
using stdexec::bulk;

#ifndef HPCG_NO_MPI
#define COMPUTE_DOT_PRODUCT(VEC1, VEC2, RESULT) \
  then([&](){ dot_local_result = 0.0; }) \
  | bulk(stdexec::par, nrow, [&](local_int_t i){ \
    dot_local_result.fetch_add((VEC1).values[i]*(VEC2).values[i], std::memory_order_relaxed); }) \
  | then([&](){ \
    dot_local_copy = dot_local_result.load(); \
    MPI_Allreduce(&dot_local_copy, &(RESULT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);})
#else
#define COMPUTE_DOT_PRODUCT(VEC1, VEC2, RESULT) \
  then([&](){ dot_local_result = 0.0; }) \
  | bulk(stdexec::par, nrow, [&](local_int_t i){ \
    dot_local_result.fetch_add((VEC1).values[i]*(VEC2).values[i], std::memory_order_relaxed); }) \
  | then([&](){ (RESULT) = dot_local_result.load(); })
#endif

#define WAXPBY(ALPHA, X, BETA, Y, W) \
  bulk(stdexec::par, nrow, [&](local_int_t i){ (W).values[i] = (ALPHA)*(X).values[i] + (BETA)*(Y).values[i]; })

#ifndef HPCG_NO_MPI
#define SPMV(A, x, y) \
  then([&](){ ExchangeHalo((A), (x)); }) \
  | stdexec::bulk(stdexec::par, (A).localNumberOfRows, [&](local_int_t i){ \
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
  stdexec::bulk(stdexec::par, (A).localNumberOfRows, [&](local_int_t i){ \
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
    for(int j = 0; j < currentNumberOfNonzeros; j++){ \
      local_int_t curCol = currentColIndices[j]; \
      sum -= currentValues[j] * xv[curCol]; \
    } \
    sum += xv[i]*currentDiagonal; \
    xv[i] = sum/currentDiagonal; \
  } \
  for(local_int_t i = nrow_SYMGS - 1; i >= 0; i--){ \
    const double * const currentValues = (A).matrixValues[i]; \
    const local_int_t * const currentColIndices = (A).mtxIndL[i]; \
    const int currentNumberOfNonzeros = (A).nonzerosInRow[i]; \
    const double  currentDiagonal = matrixDiagonal[i][0]; \
    double sum = rv[i]; \
    for(int j = 0; j < currentNumberOfNonzeros; j++){ \
      local_int_t curCol = currentColIndices[j]; \
      sum -= currentValues[j]*xv[curCol]; \
    } \
    sum += xv[i]*currentDiagonal; \
    xv[i] = sum/currentDiagonal; \
  }

#define RESTRICTION(A, rf) \
  stdexec::bulk(stdexec::par, (A).mgData->rc->localLength, \
    [&](int i){ \
    double * Axfv = (A).mgData->Axf->values; \
    double * rfv = (rf).values; \
    double * rcv = (A).mgData->rc->values; \
    local_int_t * f2c = (A).mgData->f2cOperator; \
    rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]]; \
  })

#define PROLONGATION(Af, xf) \
  stdexec::bulk(stdexec::par, (Af).mgData->rc->localLength, \
    [&](int i){ \
    double * xfv = (xf).values; \
    double * xcv = (Af).mgData->xc->values; \
    local_int_t * f2c = (Af).mgData->f2cOperator; \
    xfv[f2c[i]] += xcv[i]; \
  })

//NOTE - OMITTED MPI HALOEXCHANGE IN SYMGS
#define PRE_RECURSION_MG(A, r, x) \
  then([&](){ \
    ZeroVector((x)); \
    SYMGS((A), (r), (x)) \
  }) \
  | SPMV((A), (x), *((A).mgData->Axf)) \
  | RESTRICTION((A), (r))

#define POST_RECURSION_MG(A, r, x) \
  PROLONGATION((A), (x)) \
  | then([&](){ \
    SYMGS((A), (r), (x)) \
  })

#define TERMINAL_MG(A, r, x) \
  then([&](){ ZeroVector((x)); \
    SYMGS((A), (r), (x)) \
  })
  
#define COMPUTE_MG() \
  PRE_RECURSION_MG(*matrix_ptrs[0], *res_ptrs[0], *zval_ptrs[0]) \
  | PRE_RECURSION_MG(*matrix_ptrs[1], *res_ptrs[1], *zval_ptrs[1]) \
  | PRE_RECURSION_MG(*matrix_ptrs[2], *res_ptrs[2], *zval_ptrs[2]) \
  | TERMINAL_MG(*matrix_ptrs[3], *res_ptrs[3], *zval_ptrs[3]) \
  | POST_RECURSION_MG(*matrix_ptrs[2], *res_ptrs[2], *zval_ptrs[2]) \
  | POST_RECURSION_MG(*matrix_ptrs[1], *res_ptrs[1], *zval_ptrs[1]) \
  | POST_RECURSION_MG(*matrix_ptrs[0], *res_ptrs[0], *zval_ptrs[0])

auto CG_stdexec(auto scheduler, const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
  const int max_iter, const double tolerance, int & niters, double & normr,  double & normr0,
  double * times, bool doPreconditioning){

  double t_begin = mytimer();  //Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; //Residual vector
  Vector & z = data.z; //Preconditioned residual vector
  Vector & p = data.p; //Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;
  std::atomic<double> dot_local_result(0.0); //for summing within dot product
  double dot_local_copy; //for passing into MPIAllReduce within dot product

  //variables needed for MG computation
  std::vector<const SparseMatrix*> matrix_ptrs(4);
  std::vector<const Vector*> res_ptrs(4);
  std::vector<Vector*> zval_ptrs(4);
  matrix_ptrs[0] = &A;
  res_ptrs[0] = &r;
  zval_ptrs[0] = &z;
  for(int cnt = 1; cnt < 4; cnt++){
    matrix_ptrs[cnt] = matrix_ptrs[cnt - 1]->Ac;
    res_ptrs[cnt] = matrix_ptrs[cnt - 1]->mgData->rc;
    zval_ptrs[cnt] = matrix_ptrs[cnt - 1]->mgData->xc;
  }

  //used in SYMGS - declare here to avoid redeclarations
  local_int_t nrow_SYMGS;
  double **matrixDiagonal;
  double *rv;
  double *xv;

  if (!doPreconditioning && A.geom->rank == 0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
#endif

  sender auto pre_loop_work = schedule(scheduler) | then([&](){
    //p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x, p);
  })
  //SPMV: Ap = A*p
  | SPMV(A, p, Ap)
  //WAXPBY: r = b - Ax (x stored in p)
  | WAXPBY(1, b, -1, Ap, r)
  | COMPUTE_DOT_PRODUCT(r, r, normr)
  | then([&](){
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif
    //Record initial residual for convergence testing
    normr0 = normr;
  });

  sync_wait(std::move(pre_loop_work));
  int k = 1;
  
  //ITERATION FOR FIRST LOOP
  //FIND A MORE ELEGANT WAY OF DOING THIS!
  sender auto first_loop = schedule(scheduler)
    //NOTE - MUST FIND A MEANS OF MAKING PRECONDITIONING OPTIONAL!
    | COMPUTE_MG()
    | then([&](){ CopyVector(z, p); }) //Copy Mr to p
    | COMPUTE_DOT_PRODUCT(r, z, rtz) //rtz = r'*z
    //SPMV: Ap = A*p
    | SPMV(A, p, Ap)
    | COMPUTE_DOT_PRODUCT(p, Ap, pAp) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; })
    //WAXPBY: x = x + alpha*p
    | WAXPBY(1, x, alpha, p, x)
    //WAXPBY: r = r - alpha*Ap
    | WAXPBY(1, r, -alpha, Ap, r)
    | COMPUTE_DOT_PRODUCT(r, r, normr)
    | then([&](){ 
      normr = sqrt(normr);
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (1 % print_freq == 0 || 1 == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = 1;
    });

    stdexec::sync_wait(std::move(first_loop));

  //Start iterations
  //Convergence check accepts an error of no more than 6 significant digits of tolerance
  for(int k = 2; k <= max_iter && normr/normr0 > tolerance; k++){

    sender auto subsequent_loop = schedule(scheduler)
    | COMPUTE_MG()
    | then([&](){ oldrtz = rtz; })
    | COMPUTE_DOT_PRODUCT(r, z, rtz) //rtz = r'*z
    | then([&](){ beta = rtz/oldrtz; })
    //WAXPBY: p = beta*p + z
    | WAXPBY(1, z, beta, p, p)
    //SPMV: Ap = A*p
    | SPMV(A, p, Ap)
    | COMPUTE_DOT_PRODUCT(p, Ap, pAp) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; })
    //WAXPBY: x = x + alpha*p
    | WAXPBY(1, x, alpha, p, x)
    //WAXPBY: r = r - alpha*Ap
    | WAXPBY(1, r, -alpha, Ap, r)
    | COMPUTE_DOT_PRODUCT(r, r, normr)
    | then([&](){ 
      normr = sqrt(normr); 
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = k;
    });

    stdexec::sync_wait(std::move(subsequent_loop));
  }

  sender auto store_times = schedule(scheduler) | then([&](){
    //Store times
    times[1] += t1; //dot-product time
    times[2] += t2; //WAXPBY time
    times[3] += t3; //SPMV time
    times[4] += t4; //AllReduce time
    times[5] += t5; //preconditioner apply time
    times[0] += mytimer() - t_begin;  //Total time. All done...
  });

  stdexec::sync_wait(std::move(store_times));

  return 0;
}

