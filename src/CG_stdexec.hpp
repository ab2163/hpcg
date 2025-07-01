#include <fstream>
#include <cmath>

#include "CGData.hpp"
#include "hpcg.hpp"
#include "ComputeMG_stdexec.hpp"
#include "ComputeDotProduct_stdexec.hpp"
#include "ComputeWAXPBY_stdexec.hpp"

#include "CG_ref.hpp"

//Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

using stdexec::sender;
using stdexec::then;
using stdexec::schedule;
using stdexec::sync_wait;

auto CG_stdexec(auto scheduler, const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
  const int max_iter, const double tolerance, int & niters, double & normr,  double & normr0,
  double * times, bool doPreconditioning){

  CG_ref(A, data, b, x, max_iter, tolerance, niters, normr, normr0, times, doPreconditioning);

  /*
  double t_begin = mytimer();  //Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; //Residual vector
  Vector & z = data.z; //Preconditioned residual vector
  Vector & p = data.p; //Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;

  if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq > 50) print_freq = 50;
  if (print_freq < 1)  print_freq = 1;
#endif

  sender auto pre_loop_work = schedule(scheduler) | then([&](){
    //p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x, p); 
  })
  | ComputeSPMV_stdexec(&t3, A, p, Ap)
  | ComputeWAXPBY_stdexec(&t2, nrow, 1.0, b, -1.0, Ap, r)
  | ComputeDotProduct_stdexec(&t1, nrow, r, r, normr, t4)
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
    | ComputeMG_stdexec(NULL, A, r, z) //Apply preconditioner
    | then([&](){ TOCK(t5); }) //Preconditioner apply time
    | ComputeWAXPBY_stdexec(&t2, nrow, 1.0, z, 0.0, z, p) //Copy Mr to p
    | ComputeDotProduct_stdexec(&t1, nrow, r, z, rtz, t4) //rtz = r'*z
    | ComputeSPMV_stdexec(&t3, A, p, Ap) //Ap = A*p
    | ComputeDotProduct_stdexec(&t1, nrow, p, Ap, pAp, t4) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; })
    | ComputeWAXPBY_stdexec(&t2, nrow, 1.0, x, alpha, p, x) //x = x + alpha*p
    | ComputeWAXPBY_stdexec(&t2, nrow, 1.0, r, -alpha, Ap, r) //r = r - alpha*Ap
    | ComputeDotProduct_stdexec(&t1, nrow, r, r, normr, t4)
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
  for(int k = 2; k <= max_iter && normr/normr0 > tolerance * (1.0 + 1.0e-6); k++){

    sender auto subsequent_loop = schedule(scheduler) | then([&](){ TICK(); })
    | ComputeMG_stdexec(NULL, A, r, z) //Apply preconditioner
    | then([&](){ TOCK(t5); }) //Preconditioner apply time
    | (then([&](){ oldrtz = rtz; })
    | ComputeDotProduct_stdexec(&t1, nrow, r, z, rtz, t4) //rtz = r'*z
    | then([&](){ beta = rtz/oldrtz; })
    | ComputeWAXPBY_stdexec(&t2, nrow, 1.0, z, beta, p, p)) //p = beta*p + z
    | ComputeSPMV_stdexec(&t3, A, p, Ap) //Ap = A*p
    | ComputeDotProduct_stdexec(&t1, nrow, p, Ap, pAp, t4) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; })
    | ComputeWAXPBY_stdexec(&t2, nrow, 1.0, x, alpha, p, x) //x = x + alpha*p
    | ComputeWAXPBY_stdexec(&t2, nrow, 1.0, r, -alpha, Ap, r) //r = r - alpha*Ap
    | ComputeDotProduct_stdexec(&t1, nrow, r, r, normr, t4)
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
  */

  return 0;
}

