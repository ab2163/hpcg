#!/bin/bash

ulimit -s 16384

mkdir build
cd build

#Run make and pass the relevant compilation flag
case "$2" in
  hpc)
    echo "Using hpc path"
    ~/hpcg/configure ajinkya_hpc
    ;;
  laptop)
    echo "Using laptop path"
    /home/ajinkya/Documents/hpcg/configure ajinkya_laptop_linux
    ;;
  *)
    echo "Using laptop path"
    /home/ajinkya/Documents/hpcg/configure ajinkya_laptop_linux
    ;;
esac

#Run make and pass the relevant compilation flag
case "$1" in
  stdexec)
    echo "Compiling for stdexec"
    make USERFLAGS=-DSELECT_STDEXEC 2>&1 | tee build.log
    ;;
  stdpar)
    echo "Compiling for stdpar"
    make USERFLAGS=-DSELECT_STDPAR 2>&1 | tee build.log
    ;;
  baseline)
    echo "Compiling for baseline"
    make USERFLAGS=-DSELECT_BASELINE 2>&1 | tee build.log
    ;;
  *)
    echo "Compiling for baseline"
    make USERFLAGS=-DSELECT_BASELINE 2>&1 | tee build.log
    ;;
esac

cd ..
