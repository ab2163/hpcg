#!/bin/bash

cd build/bin
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=true
#Wait policy gives a 2X speedup on my laptop
#and solves the 100/33/33/33 problem
export OMP_WAIT_POLICY=active
./xhpcg
cat HPCG-Benchmark_3.1*.txt
cd ../..
