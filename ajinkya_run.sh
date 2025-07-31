#!/bin/bash

cd build/bin
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
./xhpcg
cat HPCG-Benchmark_3.1*.txt
cd ../..
