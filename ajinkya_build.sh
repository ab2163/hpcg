#!/bin/bash

mkdir build
cd build

#Path on Azure instance:
#~/hpcg/configure ajinkya_laptop_linux

#Path on laptop:
/home/ajinkya/Documents/hpcg/configure ajinkya_laptop_linux

#Run make and pass the relevant compilation flag
case "$1" in
  stdexec)
    echo "Compiling for stdexec"
    make USERFLAGS=-DSELECT_STDEXEC
    ;;
  stdpar)
    echo "Compiling for stdpar"
    make USERFLAGS=-DSELECT_STDPAR
    ;;
  *)
    echo "Compiling for baseline"
    make USERFLAGS=-DSELECT_BASELINE
    ;;
esac

cd ..
