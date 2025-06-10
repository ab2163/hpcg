#!/bin/bash

cd build/bin
export OMP_NUM_TRHEADS=4
./xhpcg
cd ../..
