#!/bin/bash

mkdir build
cd build

#Run make and pass the relevant compilation flag
case "$2" in
  azure)
    echo "Using azure path"
    ~/hpcg/configure ajinkya_laptop_linux
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
    make USERFLAGS=-DSELECT_STDEXEC
    ;;
  stdpar)
    echo "Compiling for stdpar"
    make USERFLAGS=-DSELECT_STDPAR
    ;;
  baseline)
    echo "Compiling for baseline"
    make USERFLAGS=-DSELECT_BASELINE
    ;;
  *)
    echo "Compiling for baseline"
    make USERFLAGS=-DSELECT_BASELINE
    ;;
esac

cd ..
