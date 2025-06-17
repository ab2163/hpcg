#include <thread>
#include <cassert>
#include <iostream>

#include "ComputeWAXPBY_stdexec.hpp"

int ComputeWAXPBY_stdexec(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  const double * const xv = x.values;
  const double * const yv = y.values;
  double * const wv = w.values;

  unsigned int num_threads = std::thread::hardware_concurrency();
  if(num_threads == 0) {
    std::cerr << "Unable to determine thread pool size.\n";
    std::exit(EXIT_FAILURE);
  }

  exec::static_thread_pool pool(num_threads);
  auto sched = pool.get_scheduler();
  auto start_point = stdexec::schedule(sched);

  if (alpha==1.0) {
    stdexec::sync_wait(stdexec::bulk(start_point, stdexec::par, n, [&](local_int_t i){ wv[i] = xv[i] + beta*yv[i]; }));
  } else if (beta==1.0) {
    stdexec::sync_wait(stdexec::bulk(start_point, stdexec::par, n, [&](local_int_t i){ wv[i] = alpha*xv[i] + yv[i]; }));
  } else {
    stdexec::sync_wait(stdexec::bulk(start_point, stdexec::par, n, [&](local_int_t i){ wv[i] = alpha*xv[i] + beta*yv[i]; }));
  }

  return 0;
}
