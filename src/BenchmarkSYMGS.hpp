#ifndef BENCHMARK_SYMGS_HPP
#define BENCHMARK_SYMGS_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include <chrono>
#include <string>
#include <functional>

struct BenchmarkResult {
    std::string name;
    double time_ms;
    double gflops;
    double bandwidth_gb_s;
    int iterations;
};

class SymGSBenchmark {
public:
    static BenchmarkResult benchmark_implementation(
        const std::string& name,
        std::function<int(const SparseMatrix&, const Vector&, Vector&)> impl,
        const SparseMatrix& A,
        const Vector& r,
        Vector& x,
        int warmup_iterations = 5,
        int benchmark_iterations = 50
    );
    
    static void run_all_benchmarks(const SparseMatrix& A, const Vector& r, Vector& x);
    
private:
    static double calculate_gflops(const SparseMatrix& A, double time_seconds, int iterations);
    static double calculate_bandwidth(const SparseMatrix& A, double time_seconds, int iterations);
};

#endif // BENCHMARK_SYMGS_HPP