#include "BenchmarkSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeSYMGS_par.hpp"
#include "ComputeSYMGS_stdexec.hpp"
#include "ComputeSYMGS_stdexec_advanced.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstring>

BenchmarkResult SymGSBenchmark::benchmark_implementation(
    const std::string& name,
    std::function<int(const SparseMatrix&, const Vector&, Vector&)> impl,
    const SparseMatrix& A,
    const Vector& r,
    Vector& x,
    int warmup_iterations,
    int benchmark_iterations) {
    
    // Warmup runs
    for(int i = 0; i < warmup_iterations; i++) {
        impl(A, r, x);
    }
    
    // Benchmark runs
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < benchmark_iterations; i++) {
        impl(A, r, x);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double time_seconds = duration.count() / 1e9;
    double time_ms = time_seconds * 1000.0 / benchmark_iterations;
    
    BenchmarkResult result;
    result.name = name;
    result.time_ms = time_ms;
    result.gflops = calculate_gflops(A, time_seconds, benchmark_iterations);
    result.bandwidth_gb_s = calculate_bandwidth(A, time_seconds, benchmark_iterations);
    result.iterations = benchmark_iterations;
    
    return result;
}

double SymGSBenchmark::calculate_gflops(const SparseMatrix& A, double time_seconds, int iterations) {
    // Each SYMGS iteration performs:
    // - 2 sweeps (forward + backward)
    // - For each row: 2*nnz + 1 FLOPs (multiply-add for each nonzero, plus division)
    
    double flops_per_iteration = 2.0 * A.localNumberOfRows * 27.0; // Assuming 27-point stencil
    double total_flops = flops_per_iteration * iterations;
    
    return (total_flops / time_seconds) / 1e9;
}

double SymGSBenchmark::calculate_bandwidth(const SparseMatrix& A, double time_seconds, int iterations) {
    // Memory traffic includes:
    // - Reading matrix values and indices
    // - Reading and writing solution vector
    // - Reading RHS vector
    
    size_t bytes_per_iteration = 
        A.localNumberOfNonzeros * (sizeof(double) + sizeof(local_int_t)) +  // Matrix data
        A.localNumberOfRows * sizeof(double) * 3;  // 2 reads + 1 write to x, 1 read from r
    
    double total_bytes = bytes_per_iteration * iterations;
    
    return (total_bytes / time_seconds) / 1e9;
}

void SymGSBenchmark::run_all_benchmarks(const SparseMatrix& A, const Vector& r, Vector& x) {
    std::cout << "\n=== Symmetric Gauss-Seidel Performance Comparison ===\n";
    std::cout << "Matrix size: " << A.localNumberOfRows << " x " << A.localNumberOfColumns << "\n";
    std::cout << "Nonzeros: " << A.localNumberOfNonzeros << "\n";
    std::cout << "Colors: 8\n\n";
    
    // Create a copy of x for each test to ensure fair comparison
    Vector x_copy;
    x_copy.localLength = x.localLength;
    x_copy.values = new double[x.localLength];
    
    std::vector<BenchmarkResult> results;
    
    // Benchmark reference implementation
    std::memcpy(x_copy.values, x.values, x.localLength * sizeof(double));
    results.push_back(benchmark_implementation(
        "Reference (Serial)", 
        ComputeSYMGS_ref, 
        A, r, x_copy
    ));
    
    // Benchmark parallel implementation
    std::memcpy(x_copy.values, x.values, x.localLength * sizeof(double));
    results.push_back(benchmark_implementation(
        "OpenMP Parallel", 
        ComputeSYMGS_par, 
        A, r, x_copy
    ));
    
    // Benchmark stdexec implementations
    std::memcpy(x_copy.values, x.values, x.localLength * sizeof(double));
    results.push_back(benchmark_implementation(
        "stdexec Standard", 
        ComputeSYMGS_stdexec, 
        A, r, x_copy
    ));
    
    std::memcpy(x_copy.values, x.values, x.localLength * sizeof(double));
    results.push_back(benchmark_implementation(
        "stdexec Pipelined", 
        ComputeSYMGS_stdexec_pipelined, 
        A, r, x_copy
    ));
    
    std::memcpy(x_copy.values, x.values, x.localLength * sizeof(double));
    results.push_back(benchmark_implementation(
        "stdexec Wavefront", 
        ComputeSYMGS_stdexec_wavefront, 
        A, r, x_copy
    ));
    
    // Print results
    std::cout << std::left << std::setw(20) << "Implementation" 
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GFLOP/s"
              << std::setw(15) << "Bandwidth (GB/s)"
              << std::setw(10) << "Speedup" << "\n";
    std::cout << std::string(69, '-') << "\n";
    
    double reference_time = results[0].time_ms;
    
    for(const auto& result : results) {
        double speedup = reference_time / result.time_ms;
        
        std::cout << std::left << std::setw(20) << result.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(12) << std::setprecision(2) << result.gflops
                  << std::setw(15) << std::setprecision(2) << result.bandwidth_gb_s
                  << std::setw(10) << std::setprecision(2) << speedup << "x\n";
    }
    
    // Find and highlight best performer
    auto best_result = std::min_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.time_ms < b.time_ms;
        });
    
    std::cout << "\nBest performer: " << best_result->name 
              << " (" << std::fixed << std::setprecision(2) 
              << reference_time / best_result->time_ms << "x speedup)\n";
    
    delete[] x_copy.values;
}