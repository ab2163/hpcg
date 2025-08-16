#!/bin/bash

ulimit -s 16384

mkdir build
cd build

#CHANGE THIS LINE TO POINT TO THE "CONFIGURE" FILE IN THE HPCG REPO:
/home/ajinkya/Documents/hpcg/configure bris

#Run make and pass the relevant compilation flag
case "$1" in
   baseline)
    echo "Compiling for baseline implementation."
    make USERFLAGS=-DSELECT_BASELINE 2>&1 | tee build.log
    ;;
   baseline_par)
    echo "Compiling for baseline implementation with parallel SYMGS."
    make USERFLAGS=-DPARALLEL_SYMGS 2>&1 | tee build.log
    ;;
  stdpar_cpu)
    echo "Compiling for stdpar CPU implementation."
    make USERFLAGS="-DSELECT_STDPAR -DPARALLEL_SYMGS -stdpar=multicore" 2>&1 | tee build.log
    ;;
  stdexec_cpu)
    echo "Compiling for stdexec CPU implementation."
    make USERFLAGS="-DSELECT_STDEXEC -DPARALLEL_SYMGS -stdpar=multicore" 2>&1 | tee build.log
    ;;
  stdpar_gpu)
    echo "Compiling for stdpar GPU implementation."
    make USERFLAGS="-DSELECT_STDPAR -DPARALLEL_SYMGS -stdpar=gpu -DUSE_GPU" 2>&1 | tee build.log
    ;;
  stdexec_gpu)
    echo "Compiling for stdexec GPU implementation."
    make USERFLAGS="-DSELECT_STDEXEC -DPARALLEL_SYMGS -stdpar=gpu -DUSE_GPU" 2>&1 | tee build.log
    ;;
  *)
    echo "Please specify a valid code implementation to build."
    ;;
esac

cd ..
