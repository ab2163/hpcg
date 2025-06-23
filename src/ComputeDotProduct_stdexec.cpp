#include <execution>
#include <atomic>
#include <ranges>

#include "ComputeDotProduct_stdexec.hpp"
#include "mytimer.hpp"

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

template <stdexec::sender Sender>
auto ComputeDotProduct_stdexec(Sender input, double & time, const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce) -> declype(stdexec::then(input, [](){})){

  return stdexec::then(input, [&](){

    double t_begin = mytimer();
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
    // Use MPI's reduce function to collect all partial sums
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

    time += mytimer() - t_begin;
  });
}
