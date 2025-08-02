//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
// Optimized stdexec implementation of Symmetric Gauss-Seidel
//
// ***************************************************
//@HEADER

#ifndef COMPUTESYMGS_STDEXEC_HPP
#define COMPUTESYMGS_STDEXEC_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

#ifdef USE_GPU
#include <nvexec/stream_context.cuh>
#endif

int ComputeSYMGS_stdexec(const SparseMatrix &A, const Vector &r, Vector &x);

#endif // COMPUTESYMGS_STDEXEC_HPP