#!/bin/bash

# Build script for optimized HPCG SYMGS implementations
# This script builds both the original HPCG and our test program

set -e  # Exit on any error

echo "=== Building Optimized SYMGS for HPCG ==="

# Check if stdexec is available
STDEXEC_PATH=""
if [ -d "/usr/local/include/stdexec" ]; then
    STDEXEC_PATH="/usr/local"
elif [ -d "/opt/stdexec/include/stdexec" ]; then
    STDEXEC_PATH="/opt/stdexec"
elif [ -n "$STDEXEC_ROOT" ]; then
    STDEXEC_PATH="$STDEXEC_ROOT"
else
    echo "Warning: stdexec not found in standard locations."
    echo "Please set STDEXEC_ROOT environment variable or install stdexec."
    echo "Falling back to OpenMP parallel version..."
    USE_STDEXEC=OFF
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring build..."
if [ -n "$STDEXEC_PATH" ] && [ "$USE_STDEXEC" != "OFF" ]; then
    echo "Using stdexec from: $STDEXEC_PATH"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DHPCG_ENABLE_STDEXEC=ON \
        -DHPCG_ENABLE_OPENMP=ON \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_PREFIX_PATH="$STDEXEC_PATH" \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -funroll-loops"
else
    echo "Building with OpenMP parallel version only..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DHPCG_ENABLE_OPENMP=ON \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -funroll-loops"
fi

# Build
echo "Building..."
make -j$(nproc)

# Build test programs
echo "Building test programs..."
cd ..
if [ -n "$STDEXEC_PATH" ] && [ "$USE_STDEXEC" != "OFF" ]; then
    # Build main test program
    g++ -std=c++20 -O3 -march=native -mtune=native -funroll-loops \
        -I. -I"$STDEXEC_PATH/include" \
        -fopenmp -DSELECT_STDEXEC \
        test_symgs_optimization.cpp \
        build/CMakeFiles/xhpcg.dir/src/*.o \
        -o test_symgs_optimization \
        -lm -lpthread
    
    # Build single-bulk focused test
    g++ -std=c++20 -O3 -march=native -mtune=native -funroll-loops \
        -I. -I"$STDEXEC_PATH/include" \
        -fopenmp -DSELECT_STDEXEC \
        test_single_bulk_symgs.cpp \
        build/CMakeFiles/xhpcg.dir/src/*.o \
        -o test_single_bulk_symgs \
        -lm -lpthread
else
    # Build main test program
    g++ -std=c++17 -O3 -march=native -mtune=native -funroll-loops \
        -I. -fopenmp -DPARALLEL_SYMGS \
        test_symgs_optimization.cpp \
        build/CMakeFiles/xhpcg.dir/src/*.o \
        -o test_symgs_optimization \
        -lm -lpthread
    
    # Build single-bulk focused test (fallback to parallel version)
    g++ -std=c++17 -O3 -march=native -mtune=native -funroll-loops \
        -I. -fopenmp -DPARALLEL_SYMGS \
        test_single_bulk_symgs.cpp \
        build/CMakeFiles/xhpcg.dir/src/*.o \
        -o test_single_bulk_symgs \
        -lm -lpthread
fi

echo "Build completed successfully!"
echo ""
echo "Executables created:"
echo "  - build/xhpcg              : Main HPCG benchmark"
echo "  - test_symgs_optimization  : SYMGS optimization test program"
echo "  - test_single_bulk_symgs   : Single-bulk SYMGS validation"
echo ""
echo "To run the tests:"
echo "  ./test_symgs_optimization [nx] [ny] [nz]     # Comprehensive comparison"
echo "  ./test_single_bulk_symgs [nx] [ny] [nz]      # Single-bulk validation"
echo "  Example: ./test_single_bulk_symgs 64 64 64"
echo ""
echo "To run the full HPCG benchmark:"
echo "  cd build && ./xhpcg"