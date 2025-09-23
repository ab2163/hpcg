//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
// Advanced pipelined stdexec implementation of Symmetric Gauss-Seidel
//
// ***************************************************
//@HEADER

#ifndef COMPUTESYMGS_STDEXEC_ADVANCED_HPP
#define COMPUTESYMGS_STDEXEC_ADVANCED_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/repeat_n.hpp>

#ifdef USE_GPU
#include <nvexec/stream_context.cuh>
#endif

// Advanced features
int ComputeSYMGS_stdexec_pipelined(const SparseMatrix &A, const Vector &r, Vector &x);
int ComputeSYMGS_stdexec_wavefront(const SparseMatrix &A, const Vector &r, Vector &x);

#endif // COMPUTESYMGS_STDEXEC_ADVANCED_HPP