//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark  
// Optimized stdexec implementation of Symmetric Gauss-Seidel
//
// ***************************************************
//@HEADER

#include "ComputeSYMGS_stdexec.hpp"
#include <cassert>
#include <algorithm>
#include <execution>
#include <numeric>
#include <ranges>

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

using namespace stdexec;

// Optimized color-based SYMGS sweep with better cache utilization
template<typename Scheduler>
auto make_color_sweep_sender(Scheduler&& sched, 
                           const SparseMatrix& A,
                           const Vector& r, 
                           Vector& x,
                           int color,
                           bool reverse_order = false) {
    
    const local_int_t nrow = A.localNumberOfRows;
    const double* const* const matrixDiagonal = A.matrixDiagonal;
    const double* const rv = r.values;
    double* const xv = x.values;
    
    return schedule(sched) | bulk(std::execution::par_unseq, nrow, 
        [=](local_int_t i) {
            // Process in reverse order for backward sweep
            local_int_t row_idx = reverse_order ? (nrow - 1 - i) : i;
            
            if(A.colors[row_idx] != color) return;
            
            const double* const currentValues = A.matrixValues[row_idx];
            const local_int_t* const currentColIndices = A.mtxIndL[row_idx];
            const int currentNumberOfNonzeros = A.nonzerosInRow[row_idx];
            const double currentDiagonal = matrixDiagonal[row_idx][0];
            
            // Use FMA for better numerical precision and performance
            double sum = rv[row_idx];
            
            // Unroll small loops for better performance on common stencil sizes
            int j = 0;
            for(; j < currentNumberOfNonzeros - 3; j += 4) {
                sum = std::fma(-currentValues[j], xv[currentColIndices[j]], sum);
                sum = std::fma(-currentValues[j+1], xv[currentColIndices[j+1]], sum);
                sum = std::fma(-currentValues[j+2], xv[currentColIndices[j+2]], sum);
                sum = std::fma(-currentValues[j+3], xv[currentColIndices[j+3]], sum);
            }
            
            // Handle remaining elements
            for(; j < currentNumberOfNonzeros; j++) {
                sum = std::fma(-currentValues[j], xv[currentColIndices[j]], sum);
            }
            
            // Remove diagonal contribution and solve
            sum = std::fma(xv[row_idx], currentDiagonal, sum);
            xv[row_idx] = sum / currentDiagonal;
        });
}

// Optimized version using color range processing for better cache locality
template<typename Scheduler>
auto make_range_based_sweep_sender(Scheduler&& sched,
                                 const SparseMatrix& A,
                                 const Vector& r,
                                 Vector& x,
                                 int color,
                                 bool reverse_order = false) {
    
    // Use the start/end indices for better cache locality
    local_int_t start_idx = A.startInds[color];
    local_int_t end_idx = A.endInds[color];
    local_int_t color_size = end_idx - start_idx;
    
    if(color_size == 0) {
        return schedule(sched) | then([](){});
    }
    
    const double* const* const matrixDiagonal = A.matrixDiagonal;
    const double* const rv = r.values;
    double* const xv = x.values;
    
    return schedule(sched) | bulk(std::execution::par_unseq, color_size,
        [=](local_int_t idx) {
            // Process in reverse order for backward sweep
            local_int_t actual_idx = reverse_order ? 
                (start_idx + color_size - 1 - idx) : (start_idx + idx);
            
            const double* const currentValues = A.matrixValues[actual_idx];
            const local_int_t* const currentColIndices = A.mtxIndL[actual_idx];
            const int currentNumberOfNonzeros = A.nonzerosInRow[actual_idx];
            const double currentDiagonal = matrixDiagonal[actual_idx][0];
            
            double sum = rv[actual_idx];
            
            // Vectorized loop with unrolling
            int j = 0;
            for(; j < currentNumberOfNonzeros - 3; j += 4) {
                sum = std::fma(-currentValues[j], xv[currentColIndices[j]], sum);
                sum = std::fma(-currentValues[j+1], xv[currentColIndices[j+1]], sum);
                sum = std::fma(-currentValues[j+2], xv[currentColIndices[j+2]], sum);
                sum = std::fma(-currentValues[j+3], xv[currentColIndices[j+3]], sum);
            }
            
            for(; j < currentNumberOfNonzeros; j++) {
                sum = std::fma(-currentValues[j], xv[currentColIndices[j]], sum);
            }
            
            sum = std::fma(xv[actual_idx], currentDiagonal, sum);
            xv[actual_idx] = sum / currentDiagonal;
        });
}

int ComputeSYMGS_stdexec(const SparseMatrix &A, const Vector &r, Vector &x) {
    assert(x.localLength == A.localNumberOfColumns);
    
#ifndef HPCG_NO_MPI
    ExchangeHalo(A, x);
#endif

    // Create scheduler based on available hardware
#ifdef USE_GPU
    nvexec::stream_context ctx;
    auto scheduler = ctx.get_scheduler();
#else
    // Use optimal thread count (usually hardware concurrency)
    static exec::static_thread_pool pool(std::thread::hardware_concurrency());
    auto scheduler = pool.get_scheduler();
#endif

    constexpr int NUM_COLORS = 8;
    constexpr int FORWARD_AND_BACKWARD = 2;
    
    // Perform the symmetric Gauss-Seidel iteration
    for(int sweep = 0; sweep < FORWARD_AND_BACKWARD; sweep++) {
        bool is_backward_sweep = (sweep == 1);
        
        // Process all colors in sequence (dependency requirement)
        for(int color = 0; color < NUM_COLORS; color++) {
            // Choose between range-based or full-scan based on color density
            local_int_t color_range = A.endInds[color] - A.startInds[color];
            local_int_t total_rows = A.localNumberOfRows;
            
            if(color_range > 0 && color_range < total_rows / 2) {
                // Use range-based processing for sparse colors
                auto work = make_range_based_sweep_sender(scheduler, A, r, x, color, is_backward_sweep);
                sync_wait(std::move(work));
            } else if(color_range > 0) {
                // Use full-scan for dense colors
                auto work = make_color_sweep_sender(scheduler, A, r, x, color, is_backward_sweep);
                sync_wait(std::move(work));
            }
        }
    }
    
    return 0;
}