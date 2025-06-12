#!/bin/bash

mkdir build
cd build

#Path on Azure instance:
#~/hpcg/configure ajinkya_laptop_linux

#Path on laptop:
/home/ajinkya/Documents/hpcg/configure ajinkya_laptop_linux
make
cd ..
