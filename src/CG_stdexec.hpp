#include <fstream>
#include <cmath>

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "hpcg.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_stdexec.hpp"
#include "ComputeMG_stdexec.hpp"
#include "ComputeDotProduct_stdexec.hpp"
#include "ComputeWAXPBY_stdexec.hpp"

//Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

using stdexec::sender;
using stdexec::then;

template <sender Sender>
decltype(auto) CG_stdexec(Sender input, const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
  const int max_iter, const double tolerance, int & niters, double & normr,  double & normr0,
  double * times, bool doPreconditioning){

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

  sender auto step1 = input | then([&](){
    //p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x, p); 
  });

  sender auto step2 = ComputeSPMV_stdexec(step1, t3, A, p, Ap);
  sender auto step3 = ComputeWAXPBY_stdexec(step2, t2, nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized);
  sender auto step4 = ComputeDotProduct_stdexec(step3, t1, nrow, r, r, normr, t4, A.isDotProductOptimized);

  sender auto step5 = then(step4, [&](){
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif
    //Record initial residual for convergence testing
    normr0 = normr;
  });
    
  //Start iterations
  //Convergence check accepts an error of no more than 6 significant digits of tolerance
  for(int k = 1; k <= max_iter && normr/normr0 > tolerance * (1.0 + 1.0e-6); k++){
    sender auto step6 = then((k > 1) ? step 20 : step5, [&](){ TICK(); });
    if(doPreconditioning)
    sender auto step7 = ComputeMG_stdexec(step6, dummy, A, r, z); //Apply preconditioner
    else
      sender auto step7 = then(step6, [&](){ CopyVector(r, z); }); //copy r to z (no preconditioning)
    sender auto step8 = then(step7, [&](){ TOCK(t5); }); //Preconditioner apply time

    if(k == 1){
      sender auto step9 = ComputeWAXPBY_stdexec(step8, t2, nrow, 1.0, z, 0.0, z, p, A.isWaxpbyOptimized); //Copy Mr to p
      sender auto step12 = ComputeDotProduct_stdexec(step9, t1, nrow, r, z, rtz, t4, A.isDotProductOptimized); //rtz = r'*z
    }else{
      sender auto step9 = then(step8, [&](){ oldrtz = rtz; });
      sender auto step10 = ComputeDotProduct_stdexec(step9, t1, nrow, r, z, rtz, t4, A.isDotProductOptimized); //rtz = r'*z
      sender auto step11 = then(step10, [&](){ beta = rtz/oldrtz; });
      sender auto step12 = ComputeWAXPBY_stdexec(step11, t2, nrow, 1.0, z, beta, p, p, A.isWaxpbyOptimized); //p = beta*p + z
    }

    sender auto step13 = ComputeSPMV_stdexec(step12, t3, A, p, Ap); //Ap = A*p
    sender auto step14 = ComputeDotProduct_stdexec(step13, t1, nrow, p, Ap, pAp, t4, A.isDotProductOptimized); //alpha = p'*Ap
    sender auto step15 = then(step14, [&](){ alpha = rtz/pAp; });
    sender auto step16 = ComputeWAXPBY_stdexec(step15, t2, nrow, 1.0, x, alpha, p, x, A.isWaxpbyOptimized); //x = x + alpha*p
    sender auto step17 = ComputeWAXPBY_stdexec(step16, t2, nrow, 1.0, r, -alpha, Ap, r, A.isWaxpbyOptimized); //r = r - alpha*Ap
    sender auto step18 = ComputeDotProduct_stdexec(step17, t1, nrow, r, r, normr, t4, A.isDotProductOptimized);
    sender auto step19 = then(step18, [&](){ normr = sqrt(normr); });
    
    sender auto step20 = then(step19, [&](){
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = k;
    });

    //because I am passing an lvalue reference to sync_wait
    //I will be able to resuse current afterwards
    stdexec::sync_wait(step20);
  }

  sender auto step21 = then(step20, [&](){
    //Store times
    times[1] += t1; //dot-product time
    times[2] += t2; //WAXPBY time
    times[3] += t3; //SPMV time
    times[4] += t4; //AllReduce time
    times[5] += t5; //preconditioner apply time

    times[0] += mytimer() - t_begin;  //Total time. All done...
  });

  return step20;
}

