#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>

#ifndef EXECUTION_INCLUDED
#define EXECUTION_INCLUDED
#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/stdexec/__detail/__senders_core.hpp"
#endif

#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED
#include "Vector.hpp"
#endif

#ifndef TIMER_INCLUDED
#define TIMER_INCLUDED
#include "mytimer.hpp"
#endif

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

auto ComputeDotProduct_stdexec(double * time, const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce){

  return stdexec::then([&, time, n](){

    if(time != NULL) *time -= mytimer();
    assert(x.localLength >= n); //Test vector lengths
    assert(y.localLength >= n);

    std::atomic<double> local_result(0.0);
    auto rows = std::views::iota(local_int_t{0}, n);
    double * xv = x.values;
    double * yv = y.values;

    if (yv == xv){
      std::for_each_n(std::execution::par, xv, n, 
        [&](double xval){ local_result.fetch_add(xval*xval, std::memory_order_relaxed); });
    }
    else{
      std::for_each_n(std::execution::par, rows.begin(), n, 
        [&](local_int_t ind){ local_result.fetch_add(xv[ind]*yv[ind], std::memory_order_relaxed); });
    }
    
#ifndef HPCG_NO_MPI
    //Use MPI's reduce function to collect all partial sums
    double t0 = mytimer();
    double global_result = 0.0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
        MPI_COMM_WORLD);
    result = global_result;
    time_allreduce += mytimer() - t0;
#else
    time_allreduce += 0.0;
    result = local_result;
#endif
    if(time != NULL) *time += mytimer();
  });
}
