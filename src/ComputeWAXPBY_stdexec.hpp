#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "Vector.hpp"

template <stdexec::sender Sender>
auto ComputeWAXPBY_stdexec(Sender input, double & time, const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w) -> decltype(stdexec::then(input, [](){}));