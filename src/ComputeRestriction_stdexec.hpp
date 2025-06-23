#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "Vector.hpp"
#include "SparseMatrix.hpp"

template <stdexec::sender Sender>
auto ComputeRestriction_stdexec(Sender input, const SparseMatrix & A, const Vector & rf)
  -> declype(stdexec::then(input, [](){}));