//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
// Single-bulk stdexec implementation of Symmetric Gauss-Seidel
// Reduces sync_wait calls from 16 to 1 per iteration
//
// ***************************************************
//@HEADER

#ifndef COMPUTESYMGS_STDEXEC_SINGLE_BULK_HPP
#define COMPUTESYMGS_STDEXEC_SINGLE_BULK_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

#ifdef USE_GPU
#include <nvexec/stream_context.cuh>
#endif

// Single bulk call implementations
int ComputeSYMGS_stdexec_single_bulk(const SparseMatrix &A, const Vector &r, Vector &x);
int ComputeSYMGS_stdexec_chunked(const SparseMatrix &A, const Vector &r, Vector &x);
int ComputeSYMGS_stdexec_wavefront_single(const SparseMatrix &A, const Vector &r, Vector &x);

#endif // COMPUTESYMGS_STDEXEC_SINGLE_BULK_HPP