#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>
#include <numeric>
#include <iostream>
#include <omp.h>
#include <cmath>

#ifdef USE_GPU
#include "../stdexec/include/nvexec/stream_context.cuh"
#endif

#include "../stdexec/include/exec/static_thread_pool.hpp"
#include "../stdexec/include/stdexec/execution.hpp"

using stdexec::sender;
using stdexec::then;
using stdexec::schedule;
using stdexec::sync_wait;
using stdexec::bulk;
using stdexec::just;
using stdexec::continues_on;

#define NUM_TH 2

int main(void){
  exec::static_thread_pool pool(NUM_TH);

#ifdef USE_GPU
  //scheduler for GPU execution
  nvexec::stream_context ctx;
  auto scheduler = ctx.get_scheduler();
#else
  //scheduler for CPU execution
  auto scheduler = pool.get_scheduler();
#endif

  int *cpu_ctr = new int;
  *cpu_ctr = 0;

  int *gpu_ctr = new int;
  *gpu_ctr = 0;

  sync_wait(schedule(scheduler) | then([=](){ (*gpu_ctr)++; }));
  std::cout << *gpu_ctr << "\n";
  //sender auto A = schedule(scheduer) | split();

  delete cpu_ctr;
  delete gpu_ctr;
  return 0;
}