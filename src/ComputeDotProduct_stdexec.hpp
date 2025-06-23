#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#include "Vector.hpp"

template <stdexec::sender Sender>
auto ComputeDotProduct_stdexec(Sender input, double & time, const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce) -> decltype(stdexec::then(input, [](){}));