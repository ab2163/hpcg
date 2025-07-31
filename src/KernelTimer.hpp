#ifndef KERNEL_TIMER_HPP
#define KERNEL_TIMER_HPP

#include <chrono>
#include <iostream>
#include <array>
#include <stdexcept>

// Enum for kernel identifiers
enum KernelID {
    SYMGS = 0,
    SPMV,
    WAXPBY,
    DOT_PRODUCT,
    PROLONGATION,
    RESTRICTION,
    NUM_KERNELS  // Total count
};

class KernelTimer {
public:
    void start(KernelID kernel);
    void stop(KernelID kernel);
    void print() const;

private:
    std::array<std::chrono::high_resolution_clock::time_point, NUM_KERNELS> start_times{};
    std::array<double, NUM_KERNELS> total_times{};
    std::array<bool, NUM_KERNELS> active{};

    void validate_kernel_id(KernelID id) const;
};

// Declare global instance
extern KernelTimer GlobalKernelTimer;

#endif // KERNEL_TIMER_HPP
