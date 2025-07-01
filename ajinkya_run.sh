#!/bin/bash

ulimit -s 16384

cd build/bin
echo "Using $(nproc) threads"
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=true
./xhpcg
cat HPCG-Benchmark_3.1*.txt
cd ../..
