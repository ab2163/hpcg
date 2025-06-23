#include <fstream>
#include <cmath>

#include "hpcg.hpp"
#include "CG_stdexec.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_stdexec.hpp"
#include "ComputeMG_stdexec.hpp"
#include "ComputeDotProduct_stdexec.hpp"
#include "ComputeWAXPBY_stdexec.hpp"

//Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

template <stdexec::sender Sender>
auto CG_stdexec(Sender input, const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
  const int max_iter, const double tolerance, int & niters, double & normr,  double & normr0,
  double * times, bool doPreconditioning) -> decltype(stdexec::then(input, [](){})){

  double t_begin = mytimer();  //Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
  double dummy = 0.0; //used to pass into functions as a placeholder
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

  Sender current = input | then([&](){
    //p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x, p); 
  });

  current = ComputeSPMV_stdexec(current, t3, A, p, Ap);
  current = ComputeWAXPBY_stdexec(current, t2, nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized);
  current = ComputeDotProduct_stdexec(current, t1, nrow, r, r, normr, t4, A.isDotProductOptimized);

  current = stdexec::then(current, [&](){
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank==0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif
    //Record initial residual for convergence testing
    normr0 = normr;
  })
    
  //Start iterations
  //Convergence check accepts an error of no more than 6 significant digits of tolerance
  for (int k = 1; k <= max_iter && normr/normr0 > tolerance * (1.0 + 1.0e-6); k++){
    current = stdexec::then(current, [&](){ TICK(); });
    if(doPreconditioning)
      current = ComputeMG_stdexec(current, dummy, A, r, z); //Apply preconditioner
    else
      current = stdexec::then(current, [&](){ CopyVector(r, z); }); //copy r to z (no preconditioning)
    current = stdexec::then(current, [&](){ TOCK(t5); }); //Preconditioner apply time

    if(k == 1){
      current = ComputeWAXPBY_stdexec(current, t2, nrow, 1.0, z, 0.0, z, p, A.isWaxpbyOptimized); //Copy Mr to p
      current = ComputeDotProduct_stdexec(current, t1, nrow, r, z, rtz, t4, A.isDotProductOptimized); //rtz = r'*z
    }else{
      current = stdexec::then(current, [&](){ oldrtz = rtz; });
      current = ComputeDotProduct_stdexec(current, t1, nrow, r, z, rtz, t4, A.isDotProductOptimized); //rtz = r'*z
      current = stdexec::then(current, [&](){ beta = rtz/oldrtz; });
      current = ComputeWAXPBY_stdexec(current, t2, nrow, 1.0, z, beta, p, p, A.isWaxpbyOptimized); //p = beta*p + z
    }

    current = ComputeSPMV_stdexec(current, t3, A, p, Ap); //Ap = A*p
    current = ComputeDotProduct_stdexec(current, t1, nrow, p, Ap, pAp, t4, A.isDotProductOptimized); //alpha = p'*Ap
    current = stdexec::then(current, [&](){ alpha = rtz/pAp; });
    current = ComputeWAXPBY_stdexec(current, t2, nrow, 1.0, x, alpha, p, x, A.isWaxpbyOptimized); //x = x + alpha*p
    current = ComputeWAXPBY_stdexec(current, t2, nrow, 1.0, r, -alpha, Ap, r, A.isWaxpbyOptimized); //r = r - alpha*Ap
    current = ComputeDotProduct_stdexec(current, t1, nrow, r, r, normr, t4, A.isDotProductOptimized);
    current = stdexec::then(current, [&](){ normr = sqrt(normr); });
    
    current = stdexec::then(current, [&](){
#ifdef HPCG_DEBUG
      if (A.geom->rank==0 && (k%print_freq == 0 || k == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = k;
    });

    //because I am passing an lvalue reference to sync_wait
    //I will be able to resuse current afterwards
    stdexec::sync_wait(current);
  }

  current = stdexec::then(current, [&](){
    //Store times
    times[1] += t1; //dot-product time
    times[2] += t2; //WAXPBY time
    times[3] += t3; //SPMV time
    times[4] += t4; //AllReduce time
    times[5] += t5; //preconditioner apply time

    times[0] += mytimer() - t_begin;  //Total time. All done...
  });

  return current;
}
