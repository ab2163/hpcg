#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template <stdexec::sender Sender>
auto ComputeMG_stdexec(Sender input, double & time, const SparseMatrix  & A, const Vector & r, Vector & x)
  -> decltype(stdexec::then(input, [](){}));