# HPCG Benchmark with C++ `stdpar` and `stdexec` Libraries

## Purpose of This Fork

This repo is a fork of the [Official HPCG Benchmark repo](https://github.com/hpcg-benchmark/hpcg). Here, we implement the HPCG benchmark using the C++ `stdpar` and experimental NVIDIA `stdexec` libraries. The `stdpar` library provides functions such as `std::for_each` and `std::transform_reduce` which allow common patterns of computation to be performed in parallel. The `stdexec` library introduces the abstractions of senders and receivers which allow the construction and execution of asynchronous task graphs. Within such graphs parallel computations may be performed using sender adaptors such as `stdexec::bulk` and `stdexec::then`.

Our HPCG implementations parallelise the symmetric Gauss-Seidel kernel using a simple multi-colouring approach with eight colours. Further, the implementations are capable of compilation for both CPU and GPU targets using the NVIDIA `nvc++` compiler.

## Repo Structure

All source code is within the `src` folder. Files associated with the `stdpar` implementation are suffixed with `_stdpar` whereas those associated with `stdexec` are suffixed `_stdexec`. An accompanying masters thesis for this project is located in the `abhalerao_thesis` folder.

## Prerequisites 

To run and build the code the following are required:

* Linux installation (since `nvc++` compiler only supports Linux)
* `nvc++` compiler version 23.3+
* MPI installation

## Build Instructions

Since `nvc++` compiler is only available for Linux, these instructions only apply to Linux-based systems. Compilation steps to produce and run the `xhpcg` executable are:

1. Clone the [GitHub repo](https://github.com/ab2163/hpcg) (using `--recursive` to also clone the `stdexec` submodule):  
   `git clone --recursive https://github.com/ab2163/hpcg`

2. Install NVIDIA `nvc++` compiler and an MPI implementation.

3. Modify the `Make.bris` file within the `setup` subdirectory of the repo such that:
   1. The `CXX` variable points to the correct location of the `nvc++` compiler e.g.:
      ```make
      CXX = /opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers/bin/nvc++
      ```
   2. The `MPinc` and `MPlib` variables point to the `include` and `lib` directories of the MPI installation e.g.:
      ```make
      MPinc = -I/usr/lib/x86_64-linux-gnu/openmpi/include
      MPlib = /usr/lib/x86_64-linux-gnu/openmpi/lib
      ```
   3. In `CXXFLAGS`, the include path for the `stdexec/include` folder within the HPCG repo is correctly set e.g.:
      ```make
      CXXFLAGS = ... -I/root/hpcg/stdexec/include
      ```

4. Change the relevant line in `bris_build.sh` (which calls the `configure` file within the HPCG repo) to match the location of the `configure` file on the system e.g.:
   ```bash
   /root/hpcg/configure bris
   ```

5. Run one of the commands below to build the code:

    | Code Implementation              | Build Command                    |
    |----------------------------------|----------------------------------|
    | Baseline                         | `bris_build.sh baseline`         |
    | Baseline with Parallel SYMGS     | `bris_build.sh baseline_par`     |
    | `stdpar` (CPU execution)         | `bris_build.sh stdpar_cpu`       |
    | `stdpar` (GPU execution)         | `bris_build.sh stdpar_gpu`       |
    | `stdexec` (CPU execution)        | `bris_build.sh stdexec_cpu`      |
    | `stdexec` (GPU execution)        | `bris_build.sh stdexec_gpu`      |

6. Adjust the environment variables in `bris_run.sh`:

    * `OMP_NUM_THREADS` should equal the number of available hardware cores
    * `OMP_PROC_BIND=spread`
    * `OMP_PLACES=cores` for `stdexec` implementation and `OMP_PLACES=threads` otherwise

7. Adjust the HPCG problem size in the `hpcg.dat` file within the `build/bin` folder.

8. Run the `bris_run.sh` script, after which an output file will be produced in the `build/bin` folder containing the benchmark results.

9. To remove all files generated during building and execution of the program, run `bris_clean.sh`.

## Acknowledgements

Thanks to Dr Tom Deakin and the University of Bristol HPC team for their support.
