//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
// Advanced single-bulk stdexec implementation with optimal memory locality
//
// ***************************************************
//@HEADER

#ifndef COMPUTESYMGS_STDEXEC_OPTIMIZED_BULK_HPP
#define COMPUTESYMGS_STDEXEC_OPTIMIZED_BULK_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

#ifdef USE_GPU
#include <nvexec/stream_context.cuh>
#endif

// Optimized single-bulk implementations
int ComputeSYMGS_stdexec_interleaved(const SparseMatrix &A, const Vector &r, Vector &x);
int ComputeSYMGS_stdexec_cache_blocked(const SparseMatrix &A, const Vector &r, Vector &x);
int ComputeSYMGS_stdexec_adaptive(const SparseMatrix &A, const Vector &r, Vector &x);

#endif // COMPUTESYMGS_STDEXEC_OPTIMIZED_BULK_HPP