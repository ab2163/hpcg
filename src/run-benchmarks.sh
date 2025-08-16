#!/bin/bash

# Number of repetitions
REPEATS=100

for ((i=1; i<=REPEATS; i++)); do
    ./kernel-test-stdpar
    ./kernel-test-stdexec
done
