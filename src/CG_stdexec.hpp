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


  if (!doPreconditioning && A.geom->rank == 0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
#endif

  sender auto pre_loop_work = schedule(scheduler) | then([&](){
    //p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x, p);
  })
  //SPMV: Ap = A*p
#ifndef HPCG_NO_MPI
  | then([&](){ ExchangeHalo(A, p); })
#endif
  | stdexec::bulk(stdexec::par, A.localNumberOfRows, [&](local_int_t i){

    double sum = 0.0;
    double *cur_vals = A.matrixValues[i];
    local_int_t *cur_inds = A.mtxIndL[i];
    int cur_nnz = A.nonzerosInRow[i];
    double *xv = p.values;

    for(int j = 0; j < cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    Ap.values[i] = sum;

  })
  //WAXPBY: r = b - Ax (x stored in p)
  | bulk(stdexec::par, nrow, [&](local_int_t i){ r.values[i] = b.values[i] - Ap.values[i]; })
  //| then([&](){ ComputeDotProduct_ref(nrow, r, r, normr, t4); })
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
  sender auto first_loop = schedule(scheduler) | then([&](){ TICK(); })
    //NOTE - MUST FIND A MEANS OF MAKING PRECONDITIONING OPTIONAL!
    | then([&](){ ComputeMG_ref(A, r, z); }) //Apply preconditioner
    | then([&](){ 
      TOCK(t5); //Preconditioner apply time
      CopyVector(z, p); }) //Copy Mr to p
    | then([&](){ ComputeDotProduct_ref(nrow, r, z, rtz, t4); }) //rtz = r'*z
    //SPMV: Ap = A*p
#ifndef HPCG_NO_MPI
    | then([&](){ ExchangeHalo(A, p); })
#endif
    | stdexec::bulk(stdexec::par, A.localNumberOfRows, [&](local_int_t i){

      double sum = 0.0;
      double *cur_vals = A.matrixValues[i];
      local_int_t *cur_inds = A.mtxIndL[i];
      int cur_nnz = A.nonzerosInRow[i];
      double *xv = p.values;

      for(int j = 0; j < cur_nnz; j++)
        sum += cur_vals[j]*xv[cur_inds[j]];
      Ap.values[i] = sum;

    })
    | then([&](){ ComputeDotProduct_ref(nrow, p, Ap, pAp, t4); }) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; })
    //WAXPBY: x = x + alpha*p
    | bulk(stdexec::par, nrow, [&](local_int_t i){ x.values[i] = x.values[i] + alpha*p.values[i]; })
    //WAXPBY: r = r - alpha*Ap
    | bulk(stdexec::par, nrow, [&](local_int_t i){ r.values[i] = r.values[i] - alpha*Ap.values[i]; })
    | then([&](){ ComputeDotProduct_ref(nrow, r, r, normr, t4); })
    | then([&](){ normr = sqrt(normr); })
    | then([&](){
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

    sender auto subsequent_loop = schedule(scheduler) | then([&](){ TICK(); })
    | then([&](){ ComputeMG_ref(A, r, z); }) //Apply preconditioner
    | then([&](){ TOCK(t5); }) //Preconditioner apply time
    | then([&](){ oldrtz = rtz; })
    | then([&](){ ComputeDotProduct_ref(nrow, r, z, rtz, t4); }) //rtz = r'*z
    | then([&](){ beta = rtz/oldrtz; })
    //WAXPBY: p = beta*p + z
    | bulk(stdexec::par, nrow, [&](local_int_t i){ p.values[i] = beta*p.values[i] + z.values[i]; })
    //SPMV: Ap = A*p
#ifndef HPCG_NO_MPI
    | then([&](){ ExchangeHalo(A, p); })
#endif
    | stdexec::bulk(stdexec::par, A.localNumberOfRows, [&](local_int_t i){

      double sum = 0.0;
      double *cur_vals = A.matrixValues[i];
      local_int_t *cur_inds = A.mtxIndL[i];
      int cur_nnz = A.nonzerosInRow[i];
      double *xv = p.values;

      for(int j = 0; j < cur_nnz; j++)
        sum += cur_vals[j]*xv[cur_inds[j]];
      Ap.values[i] = sum;

    })
    | then([&](){ ComputeDotProduct_ref(nrow, p, Ap, pAp, t4); }) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; })
    //WAXPBY: x = x + alpha*p
    | bulk(stdexec::par, nrow, [&](local_int_t i){ x.values[i] = x.values[i] + alpha*p.values[i]; })
    //WAXPBY: r = r - alpha*Ap
    | bulk(stdexec::par, nrow, [&](local_int_t i){ r.values[i] = r.values[i] - alpha*Ap.values[i]; })
    | then([&](){ ComputeDotProduct_ref(nrow, r, r, normr, t4); })
    | then([&](){ normr = sqrt(normr); })
    | then([&](){
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

