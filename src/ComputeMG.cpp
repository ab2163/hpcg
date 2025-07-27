//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"

#ifdef SELECT_STDPAR
#include "ComputeMG_stdpar.hpp"
#elif defined(PARALLEL_SYMGS)
#include "ComputeMG_par.hpp"
#else
#include "ComputeMG_ref.hpp"
#endif

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  &A, const Vector &r, Vector &x){

#ifdef SELECT_STDPAR
  return ComputeMG_stdpar(A, r, x);
#elif defined(PARALLEL_SYMGS)
  return ComputeMG_par(A, r, x);
#else
  A.isMgOptimized = false;
  return ComputeMG_ref(A, r, x);
#endif
}
