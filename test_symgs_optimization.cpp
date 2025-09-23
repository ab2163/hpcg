#include "src/SparseMatrix.hpp"
#include "src/Vector.hpp"
#include "src/Geometry.hpp"
#include "src/GenerateProblem.hpp"
#include "src/BenchmarkSYMGS.hpp"
#include "src/ComputeSYMGS.hpp"
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    std::cout << "=== HPCG Symmetric Gauss-Seidel Optimization Test ===\n";
    
    // Default problem size - can be adjusted for testing
    int nx = 32, ny = 32, nz = 32;
    
    if(argc >= 4) {
        nx = std::atoi(argv[1]);
        ny = std::atoi(argv[2]);
        nz = std::atoi(argv[3]);
    }
    
    std::cout << "Problem size: " << nx << "x" << ny << "x" << nz << " = " << (nx*ny*nz) << " unknowns\n";
    
    // Initialize geometry
    Geometry geom;
    geom.size = 1;      // Single process
    geom.rank = 0;
    geom.nx = nx;
    geom.ny = ny;
    geom.nz = nz;
    geom.npx = 1;
    geom.npy = 1;
    geom.npz = 1;
    
    // Create and initialize sparse matrix
    SparseMatrix A;
    InitializeSparseMatrix(A, &geom);
    
    // Create vectors
    Vector b, x, r;
    
    // Generate the problem (this will create the matrix structure and vectors)
    std::cout << "Generating problem...\n";
    GenerateProblem(A, &b, &x, 0);  // 0 = don't create coarse grid for this test
    
    // Create residual vector with same structure as x
    r.localLength = x.localLength;
    r.values = new double[r.localLength];
    
    // Initialize with a simple test pattern
    for(local_int_t i = 0; i < x.localLength; i++) {
        x.values[i] = 1.0;  // Initial guess
        r.values[i] = static_cast<double>(i % 10) / 10.0;  // Simple test RHS
    }
    
    std::cout << "Matrix created with " << A.localNumberOfRows << " rows and " 
              << A.localNumberOfNonzeros << " nonzeros.\n";
    
    // Test basic functionality first
    std::cout << "\nTesting basic functionality...\n";
    
    // Create a copy for testing
    Vector x_test;
    x_test.localLength = x.localLength;
    x_test.values = new double[x_test.localLength];
    std::memcpy(x_test.values, x.values, x_test.localLength * sizeof(double));
    
    // Test the main ComputeSYMGS function
    int result = ComputeSYMGS(A, r, x_test);
    if(result == 0) {
        std::cout << "✓ ComputeSYMGS executed successfully\n";
    } else {
        std::cout << "✗ ComputeSYMGS failed with error code " << result << "\n";
        return 1;
    }
    
    // Show some sample values to verify computation
    std::cout << "Sample solution values after SYMGS:\n";
    for(int i = 0; i < std::min(10, static_cast<int>(x_test.localLength)); i++) {
        std::cout << "  x[" << i << "] = " << x_test.values[i] << "\n";
    }
    
    // Run comprehensive benchmarks
    std::cout << "\n" << std::string(60, '=') << "\n";
    SymGSBenchmark::run_all_benchmarks(A, r, x);
    
    // Performance recommendations
    std::cout << "\n=== Performance Optimization Recommendations ===\n";
    std::cout << "1. For your problem size (" << A.localNumberOfRows << " rows):\n";
    
    if(A.localNumberOfRows > 100000) {
        std::cout << "   - Consider using the pipelined implementation\n";
        std::cout << "   - Test with different thread pool sizes\n";
        std::cout << "   - Consider GPU acceleration if available\n";
    } else {
        std::cout << "   - Standard optimized implementation should be sufficient\n";
        std::cout << "   - Focus on memory access optimizations\n";
    }
    
    std::cout << "2. General optimizations:\n";
    std::cout << "   - Ensure proper matrix coloring for your grid\n";
    std::cout << "   - Consider memory layout optimizations\n";
    std::cout << "   - Tune thread pool size to match hardware\n";
    std::cout << "   - Profile with performance tools for hotspots\n";
    
    std::cout << "3. Advanced techniques:\n";
    std::cout << "   - Wavefront scheduling for better cache utilization\n";
    std::cout << "   - NUMA-aware memory allocation\n";
    std::cout << "   - Mixed precision for reduced memory bandwidth\n";
    
    // Cleanup
    delete[] r.values;
    delete[] x_test.values;
    // Note: Other cleanup would be handled by the problem generation cleanup
    
    std::cout << "\nTest completed successfully!\n";
    return 0;
}