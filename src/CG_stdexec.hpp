#ifndef CG_HPP
#define CG_HPP

#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"

template <stdexec::sender Sender>
auto CG_stdexec(Sender input, const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
  const int max_iter, const double tolerance, int & niters, double & normr,  double & normr0,
  double * times, bool doPreconditioning) -> declype(stdexec::then(input, [](){}));

#endif  //CG_HPP
