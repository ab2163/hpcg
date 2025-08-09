//root@ubuntu:~/hpcg/src# /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/nvc++ 
//-std=c++20 -stdpar=gpu -DUSE_GPU sndex3.cpp -o sndex3 -I/root/hpcg/stdexec/include

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
#include <nvexec/stream_context.cuh>
#endif

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

using stdexec::sender;
using stdexec::then;
using stdexec::schedule;
using stdexec::sync_wait;
using stdexec::bulk;
using stdexec::just;
using stdexec::continues_on;
using stdexec::split;
using stdexec::when_all;

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

  auto cpu_scheduler = pool.get_scheduler();

  int *cpu_ctr = new int;
  *cpu_ctr = 0;

  int *gpu_ctr = new int;
  *gpu_ctr = 0;

  //sync_wait(schedule(scheduler) | then([=](){ (*gpu_ctr)++; }));
  //std::cout << *gpu_ctr << "\n";

  //sender auto A = schedule(scheduler) | split();
  //sender auto B = schedule(cpu_scheduler) | then([=](){ (*cpu_ctr)++; });
  //sender auto C = ((schedule(cpu_scheduler) | then([](){ return; }) | continues_on(scheduler)) | then([=](){ (*gpu_ctr)++; })) | continues_on(cpu_scheduler);
  //sender auto D = when_all(B, C);
  //sync_wait(C);

  //WORKING EXAMPLE: FORK-JOIN SENDERS ON CPU
  //NOTE - COMPILE WITHOUT GPU FLAGS
  /*
  sender auto A = schedule(scheduler) | split();
  sender auto B = A | then([=](){ (*cpu_ctr)++; });
  sender auto C = A | then([=](){ (*gpu_ctr)++; });
  sender auto D = when_all(B, C);
  sync_wait(D);
  */

  sender auto A = schedule(scheduler) | split();
  sender auto B = A | then([=](){ (*cpu_ctr)++; });
  sender auto C = A | then([=](){ (*gpu_ctr)++; });
  sender auto D = when_all(B, C);
  sync_wait(D); 

  std::cout << *gpu_ctr << "\n";
  std::cout << *cpu_ctr << "\n";

  delete cpu_ctr;
  delete gpu_ctr;
  return 0;
}