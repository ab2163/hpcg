#!/bin/bash

cd build/bin
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
./xhpcg
cd ../..
