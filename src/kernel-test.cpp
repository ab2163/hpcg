#include "../stdexec/include/exec/static_thread_pool.hpp"
#include "../stdexec/include/stdexec/execution.hpp"
#include "../stdexec/include/exec/inline_scheduler.hpp"
#include "../stdexec/include/nvexec/stream_context.cuh"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <execution>
#include <numeric>
#include <ranges>  

#define ARR_SZ 100000000
using namespace std::chrono;

int main(void){
    double *long_arr = new double[ARR_SZ];
    double *transformed_arr = new double[ARR_SZ];

    //initialise the input array
    for(int ind = 0; ind < ARR_SZ; ++ind){
        long_arr[ind] = rand() % 100;
    }

    //kernel for gpu testing
    auto transform_func = [=](int ind){
        transformed_arr[ind] = long_arr[ind]*long_arr[ind] 
          + 0.5*long_arr[ARR_SZ-(ind+1)]; 
    };

    //GPU scheduler creation
    nvexec::stream_context ctx;
    auto sched_gpu = ctx.get_scheduler();

    auto indices = std::views::iota(0, ARR_SZ);
    auto start = high_resolution_clock::now();

#ifdef STDPAR
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), 
      transform_func);
#endif

#ifdef STDEXEC
    stdexec::sync_wait(stdexec::schedule(sched_gpu) 
      | stdexec::bulk(stdexec::par_unseq, ARR_SZ, transform_func));
#endif

    auto end = high_resolution_clock::now();
    auto duration_ms = duration_cast<milliseconds>(end - start).count();
    std::cout << "Elapsed time ";
#ifdef STDPAR
    std::cout << "(stdpar): ";
#endif
#ifdef STDEXEC
    std::cout << "(stdexec): ";
#endif
    std::cout << duration_ms << " ms\n";
    
    return(0);
}
