
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef COMPUTEMG_HPP
#define COMPUTEMG_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include <vector>

int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x,
  std::vector<int> & color);

#endif // COMPUTEMG_HPP
