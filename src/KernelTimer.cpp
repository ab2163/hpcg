#include <vector>
#include <algorithm>
#include <iomanip>  // for formatting
#include "KernelTimer.hpp"

KernelTimer GlobalKernelTimer;

void KernelTimer::validate_kernel_id(KernelID id) const {
    if (id < 0 || id >= NUM_KERNELS) {
        throw std::out_of_range("Invalid kernel ID.");
    }
}

void KernelTimer::start(KernelID id) {
    validate_kernel_id(id);
    if (active[id]) {
        throw std::runtime_error("Kernel already started.");
    }
    start_times[id] = std::chrono::high_resolution_clock::now();
    active[id] = true;
}

void KernelTimer::stop(KernelID id) {
    validate_kernel_id(id);
    if (!active[id]) {
        throw std::runtime_error("Kernel not started.");
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start_times[id];
    total_times[id] += elapsed.count();
    active[id] = false;
}

void KernelTimer::print() const {
    static const char* names[NUM_KERNELS] = {
        "SYMGS", "SPMV", "WAXPBY", "DOT_PRODUCT", "PROLONGATION", "RESTRICTION"
    };

    double grand_total = 0.0;
    for (double t : total_times) {
        grand_total += t;
    }

    // Build vector of (kernel_id, time) pairs
    std::vector<std::pair<int, double>> kernel_times;
    for (int i = 0; i < NUM_KERNELS; ++i) {
        kernel_times.emplace_back(i, total_times[i]);
    }

    // Sort by time descending
    std::sort(kernel_times.begin(), kernel_times.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Print
    std::cout << "=== Kernel Timing Summary (Sorted by Time) ===\n";
    for (const auto& [i, time] : kernel_times) {
        double percent = (grand_total > 0.0) ? (time / grand_total * 100.0) : 0.0;
        std::cout << std::setw(14) << std::left << names[i] << ": "
                  << std::fixed << std::setprecision(6)
                  << time << " sec ("
                  << std::setprecision(2) << percent << "%)\n";
    }
    std::cout << "Total execution time: " << std::fixed << std::setprecision(6)
              << grand_total << " sec\n";
    std::cout << "=============================================\n";
}