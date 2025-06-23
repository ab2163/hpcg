#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "Vector.hpp"
#include "SparseMatrix.hpp"

template <stdexec::sender Sender>
auto ComputeProlongation_stdexec(Sender input, const SparseMatrix & Af, Vector & xf)
  -> declype(stdexec::then(input, [](){}));