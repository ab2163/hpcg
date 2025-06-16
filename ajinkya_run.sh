#!/bin/bash

cd build/bin
export OMP_NUM_TRHEADS=4
./xhpcg
cat HPCG-Benchmark_3.1*.txt
cd ../..
