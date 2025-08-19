#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "mytimer.hpp"

#pragma once
extern double times_mg_levels[4];

int ComputeMG_stdpar(const SparseMatrix  &A, const Vector &r, Vector &x);