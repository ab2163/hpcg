#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "Vector.hpp"
#include "SparseMatrix.hpp"

template <stdexec::sender Sender>
auto ComputeSPMV_stdexec(Sender input, double & time, const SparseMatrix & A, Vector  & x, Vector & y)
  -> decltype(stdexec::then(input, [](){}));