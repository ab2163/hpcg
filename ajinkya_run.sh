#!/bin/bash

cd build/bin
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
./xhpcg
cat HPCG-Benchmark_3.1*.txt
cd ../..
