//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
// Single-bulk stdexec implementation of Symmetric Gauss-Seidel
//
// ***************************************************
//@HEADER

#include "ComputeSYMGS_stdexec_single_bulk.hpp"
#include <cassert>
#include <algorithm>
#include <execution>
#include <atomic>
#include <barrier>
#include <thread>

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

using namespace stdexec;

// Solution 1: Single bulk call with work chunks that handle multiple colors
template<typename Scheduler>
auto make_single_bulk_sender(Scheduler&& sched, 
                            const SparseMatrix& A,
                            const Vector& r,
                            Vector& x) {
    
    const local_int_t nrow = A.localNumberOfRows;
    const double* const rv = r.values;
    double* const xv = x.values;
    
    // Calculate optimal chunk size based on cache and thread count
    const int num_threads = std::thread::hardware_concurrency();
    const int chunk_size = std::max(64, nrow / (num_threads * 4));
    const int num_chunks = (nrow + chunk_size - 1) / chunk_size;
    
    return schedule(sched) | bulk(std::execution::par_unseq, num_chunks,
        [=](int chunk_id) {
            local_int_t start_row = chunk_id * chunk_size;
            local_int_t end_row = std::min(start_row + chunk_size, nrow);
            
            // Perform complete SYMGS (16 color sweeps) on this chunk
            // Forward sweep (colors 0-7) then backward sweep (colors 7-0)
            for(int sweep = 0; sweep < 2; sweep++) {
                bool is_backward = (sweep == 1);
                
                for(int color_step = 0; color_step < 8; color_step++) {
                    int color = is_backward ? (7 - color_step) : color_step;
                    
                    // Process all rows in this chunk for current color
                    for(local_int_t i = start_row; i < end_row; i++) {
                        local_int_t row_idx = is_backward ? (end_row - 1 - (i - start_row)) : i;
                        
                        if(A.colors[row_idx] != color) continue;
                        
                        const double* const currentValues = A.matrixValues[row_idx];
                        const local_int_t* const currentColIndices = A.mtxIndL[row_idx];
                        const int currentNumberOfNonzeros = A.nonzerosInRow[row_idx];
                        const double currentDiagonal = A.matrixDiagonal[row_idx][0];
                        
                        double sum = rv[row_idx];
                        
                        // Optimized inner loop with unrolling
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
                        
                        sum = std::fma(xv[row_idx], currentDiagonal, sum);
                        xv[row_idx] = sum / currentDiagonal;
                    }
                }
            }
        });
}

// Solution 2: Chunked approach with better load balancing
template<typename Scheduler>
auto make_chunked_sender(Scheduler&& sched,
                        const SparseMatrix& A,
                        const Vector& r,
                        Vector& x) {
    
    const local_int_t nrow = A.localNumberOfRows;
    const double* const rv = r.values;
    double* const xv = x.values;
    
    // Pre-compute color ranges for better load balancing
    std::array<local_int_t, 8> color_counts = {0};
    for(local_int_t i = 0; i < nrow; i++) {
        color_counts[A.colors[i]]++;
    }
    
    // Find the maximum color count to determine work units
    local_int_t max_color_work = *std::max_element(color_counts.begin(), color_counts.end());
    
    // Create work units that process rows in a round-robin fashion
    const int work_unit_size = 256; // Tunable parameter
    const int num_work_units = (max_color_work + work_unit_size - 1) / work_unit_size;
    
    return schedule(sched) | bulk(std::execution::par_unseq, num_work_units,
        [=](int work_unit_id) {
            // Each work unit processes a subset of rows for all colors
            local_int_t start_offset = work_unit_id * work_unit_size;
            local_int_t end_offset = std::min(start_offset + work_unit_size, max_color_work);
            
            // Forward and backward sweeps
            for(int sweep = 0; sweep < 2; sweep++) {
                bool is_backward = (sweep == 1);
                
                for(int color_step = 0; color_step < 8; color_step++) {
                    int color = is_backward ? (7 - color_step) : color_step;
                    
                    // Find rows belonging to this color in our work range
                    local_int_t current_offset = 0;
                    for(local_int_t i = 0; i < nrow; i++) {
                        if(A.colors[i] != color) continue;
                        
                        if(current_offset >= start_offset && current_offset < end_offset) {
                            local_int_t row_idx = i;
                            
                            const double* const currentValues = A.matrixValues[row_idx];
                            const local_int_t* const currentColIndices = A.mtxIndL[row_idx];
                            const int currentNumberOfNonzeros = A.nonzerosInRow[row_idx];
                            const double currentDiagonal = A.matrixDiagonal[row_idx][0];
                            
                            double sum = rv[row_idx];
                            
                            for(int j = 0; j < currentNumberOfNonzeros; j++) {
                                sum = std::fma(-currentValues[j], xv[currentColIndices[j]], sum);
                            }
                            
                            sum = std::fma(xv[row_idx], currentDiagonal, sum);
                            xv[row_idx] = sum / currentDiagonal;
                        }
                        
                        current_offset++;
                        if(current_offset >= end_offset) break;
                    }
                }
            }
        });
}

// Solution 3: Wavefront approach with single bulk call
template<typename Scheduler>
auto make_wavefront_single_sender(Scheduler&& sched,
                                 const SparseMatrix& A,
                                 const Vector& r,
                                 Vector& x) {
    
    const local_int_t nrow = A.localNumberOfRows;
    const double* const rv = r.values;
    double* const xv = x.values;
    
    // Calculate wavefront parameters
    constexpr int WAVEFRONT_SIZE = 128;
    const int num_wavefronts = (nrow + WAVEFRONT_SIZE - 1) / WAVEFRONT_SIZE;
    
    // Total work units: wavefronts × colors × sweeps
    const int total_work_units = num_wavefronts * 8 * 2;
    
    return schedule(sched) | bulk(std::execution::par_unseq, total_work_units,
        [=](int work_id) {
            // Decode work_id into sweep, color, and wavefront
            int sweep = work_id / (num_wavefronts * 8);
            int remaining = work_id % (num_wavefronts * 8);
            int color_step = remaining / num_wavefronts;
            int wavefront_id = remaining % num_wavefronts;
            
            bool is_backward = (sweep == 1);
            int color = is_backward ? (7 - color_step) : color_step;
            
            // Calculate wavefront bounds
            local_int_t start_row = wavefront_id * WAVEFRONT_SIZE;
            local_int_t end_row = std::min(start_row + WAVEFRONT_SIZE, nrow);
            
            // Process this wavefront for the current color
            for(local_int_t i = start_row; i < end_row; i++) {
                local_int_t row_idx = is_backward ? (end_row - 1 - (i - start_row)) : i;
                
                if(A.colors[row_idx] != color) continue;
                
                const double* const currentValues = A.matrixValues[row_idx];
                const local_int_t* const currentColIndices = A.mtxIndL[row_idx];
                const int currentNumberOfNonzeros = A.nonzerosInRow[row_idx];
                const double currentDiagonal = A.matrixDiagonal[row_idx][0];
                
                double sum = rv[row_idx];
                
                for(int j = 0; j < currentNumberOfNonzeros; j++) {
                    sum = std::fma(-currentValues[j], xv[currentColIndices[j]], sum);
                }
                
                sum = std::fma(xv[row_idx], currentDiagonal, sum);
                xv[row_idx] = sum / currentDiagonal;
            }
        });
}

int ComputeSYMGS_stdexec_single_bulk(const SparseMatrix &A, const Vector &r, Vector &x) {
    assert(x.localLength == A.localNumberOfColumns);
    
#ifndef HPCG_NO_MPI
    ExchangeHalo(A, x);
#endif

#ifdef USE_GPU
    nvexec::stream_context ctx;
    auto scheduler = ctx.get_scheduler();
#else
    static exec::static_thread_pool pool(std::thread::hardware_concurrency());
    auto scheduler = pool.get_scheduler();
#endif

    auto work = make_single_bulk_sender(scheduler, A, r, x);
    sync_wait(std::move(work)); // Only ONE sync_wait per SYMGS call!
    
    return 0;
}

int ComputeSYMGS_stdexec_chunked(const SparseMatrix &A, const Vector &r, Vector &x) {
    assert(x.localLength == A.localNumberOfColumns);
    
#ifndef HPCG_NO_MPI
    ExchangeHalo(A, x);
#endif

#ifdef USE_GPU
    nvexec::stream_context ctx;
    auto scheduler = ctx.get_scheduler();
#else
    static exec::static_thread_pool pool(std::thread::hardware_concurrency());
    auto scheduler = pool.get_scheduler();
#endif

    auto work = make_chunked_sender(scheduler, A, r, x);
    sync_wait(std::move(work)); // Only ONE sync_wait!
    
    return 0;
}

int ComputeSYMGS_stdexec_wavefront_single(const SparseMatrix &A, const Vector &r, Vector &x) {
    assert(x.localLength == A.localNumberOfColumns);
    
#ifndef HPCG_NO_MPI
    ExchangeHalo(A, x);
#endif

#ifdef USE_GPU
    nvexec::stream_context ctx;
    auto scheduler = ctx.get_scheduler();
#else
    static exec::static_thread_pool pool(std::thread::hardware_concurrency());
    auto scheduler = pool.get_scheduler();
#endif

    auto work = make_wavefront_single_sender(scheduler, A, r, x);
    sync_wait(std::move(work)); // Only ONE sync_wait!
    
    return 0;
}