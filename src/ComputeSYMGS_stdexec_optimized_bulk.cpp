//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
// Advanced single-bulk stdexec implementation with optimal memory locality
//
// ***************************************************
//@HEADER

#include "ComputeSYMGS_stdexec_optimized_bulk.hpp"
#include <cassert>
#include <algorithm>
#include <execution>
#include <vector>
#include <memory>
#include <thread>

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

using namespace stdexec;

// Cache-aware data structure for better memory locality
struct ColoredRowData {
    local_int_t row_idx;
    const double* values;
    const local_int_t* col_indices;
    int nnz;
    double diagonal;
};

// Pre-sort rows by color for better cache performance
template<typename Scheduler>
auto make_interleaved_sender(Scheduler&& sched,
                            const SparseMatrix& A,
                            const Vector& r,
                            Vector& x) {
    
    const local_int_t nrow = A.localNumberOfRows;
    const double* const rv = r.values;
    double* const xv = x.values;
    
    // Pre-compute color-sorted row data for cache efficiency
    std::array<std::vector<ColoredRowData>, 8> color_rows;
    
    for(local_int_t i = 0; i < nrow; i++) {
        int color = A.colors[i];
        color_rows[color].push_back({
            i,
            A.matrixValues[i],
            A.mtxIndL[i],
            A.nonzerosInRow[i],
            A.matrixDiagonal[i][0]
        });
    }
    
    // Calculate work distribution
    local_int_t total_color_work = 0;
    for(int c = 0; c < 8; c++) {
        total_color_work += color_rows[c].size();
    }
    
    const int num_threads = std::thread::hardware_concurrency();
    const int work_per_thread = std::max(256, static_cast<int>(total_color_work / (num_threads * 2)));
    const int num_work_units = (total_color_work + work_per_thread - 1) / work_per_thread;
    
    // Move color_rows to heap for capture
    auto color_rows_ptr = std::make_shared<std::array<std::vector<ColoredRowData>, 8>>(std::move(color_rows));
    
    return schedule(sched) | bulk(std::execution::par_unseq, num_work_units,
        [=](int work_unit_id) {
            local_int_t start_work = work_unit_id * work_per_thread;
            local_int_t end_work = std::min(start_work + work_per_thread, total_color_work);
            
            // Process forward and backward sweeps
            for(int sweep = 0; sweep < 2; sweep++) {
                bool is_backward = (sweep == 1);
                
                for(int color_step = 0; color_step < 8; color_step++) {
                    int color = is_backward ? (7 - color_step) : color_step;
                    const auto& rows = (*color_rows_ptr)[color];
                    
                    // Process rows in this color within our work range
                    local_int_t current_work = 0;
                    
                    // Calculate which rows in previous colors we need to skip
                    for(int prev_color = 0; prev_color < color; prev_color++) {
                        current_work += (*color_rows_ptr)[prev_color].size();
                    }
                    
                    if(current_work >= end_work) continue;
                    
                    local_int_t local_start = (current_work < start_work) ? start_work - current_work : 0;
                    local_int_t local_end = std::min(static_cast<local_int_t>(rows.size()), 
                                                   end_work - current_work);
                    
                    if(local_start >= local_end) continue;
                    
                    // Process rows in optimal order
                    for(local_int_t idx = local_start; idx < local_end; idx++) {
                        local_int_t actual_idx = is_backward ? (local_end - 1 - (idx - local_start)) : idx;
                        const auto& row_data = rows[actual_idx];
                        
                        double sum = rv[row_data.row_idx];
                        
                        // Optimized computation with prefetching
                        const double* values = row_data.values;
                        const local_int_t* indices = row_data.col_indices;
                        const int nnz = row_data.nnz;
                        
                        // Unrolled loop for better performance
                        int j = 0;
                        for(; j < nnz - 7; j += 8) {
                            sum = std::fma(-values[j], xv[indices[j]], sum);
                            sum = std::fma(-values[j+1], xv[indices[j+1]], sum);
                            sum = std::fma(-values[j+2], xv[indices[j+2]], sum);
                            sum = std::fma(-values[j+3], xv[indices[j+3]], sum);
                            sum = std::fma(-values[j+4], xv[indices[j+4]], sum);
                            sum = std::fma(-values[j+5], xv[indices[j+5]], sum);
                            sum = std::fma(-values[j+6], xv[indices[j+6]], sum);
                            sum = std::fma(-values[j+7], xv[indices[j+7]], sum);
                        }
                        
                        for(; j < nnz; j++) {
                            sum = std::fma(-values[j], xv[indices[j]], sum);
                        }
                        
                        sum = std::fma(xv[row_data.row_idx], row_data.diagonal, sum);
                        xv[row_data.row_idx] = sum / row_data.diagonal;
                    }
                }
            }
        });
}

// Cache-blocked approach for large problems
template<typename Scheduler>
auto make_cache_blocked_sender(Scheduler&& sched,
                              const SparseMatrix& A,
                              const Vector& r,
                              Vector& x) {
    
    const local_int_t nrow = A.localNumberOfRows;
    const double* const rv = r.values;
    double* const xv = x.values;
    
    // Cache block size - tune based on L1/L2 cache size
    constexpr int CACHE_BLOCK_SIZE = 1024; // ~8KB for double precision
    const int num_blocks = (nrow + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE;
    
    // Total work units = blocks × sweeps × colors
    const int total_work_units = num_blocks * 2 * 8;
    
    return schedule(sched) | bulk(std::execution::par_unseq, total_work_units,
        [=](int work_id) {
            // Decode work_id
            int sweep = work_id / (num_blocks * 8);
            int remaining = work_id % (num_blocks * 8);
            int color_step = remaining / num_blocks;
            int block_id = remaining % num_blocks;
            
            bool is_backward = (sweep == 1);
            int color = is_backward ? (7 - color_step) : color_step;
            
            // Calculate block boundaries
            local_int_t start_row = block_id * CACHE_BLOCK_SIZE;
            local_int_t end_row = std::min(start_row + CACHE_BLOCK_SIZE, nrow);
            
            // Process this block for the current color
            for(local_int_t i = start_row; i < end_row; i++) {
                local_int_t row_idx = is_backward ? (end_row - 1 - (i - start_row)) : i;
                
                if(A.colors[row_idx] != color) continue;
                
                const double* const currentValues = A.matrixValues[row_idx];
                const local_int_t* const currentColIndices = A.mtxIndL[row_idx];
                const int currentNumberOfNonzeros = A.nonzerosInRow[row_idx];
                const double currentDiagonal = A.matrixDiagonal[row_idx][0];
                
                double sum = rv[row_idx];
                
                // Optimized inner loop
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
        });
}

// Adaptive approach that chooses the best strategy based on problem characteristics
template<typename Scheduler>
auto make_adaptive_sender(Scheduler&& sched,
                         const SparseMatrix& A,
                         const Vector& r,
                         Vector& x) {
    
    const local_int_t nrow = A.localNumberOfRows;
    
    // Analyze color distribution to choose optimal strategy
    std::array<local_int_t, 8> color_counts = {0};
    for(local_int_t i = 0; i < nrow; i++) {
        color_counts[A.colors[i]]++;
    }
    
    // Calculate color imbalance
    auto [min_color, max_color] = std::minmax_element(color_counts.begin(), color_counts.end());
    double imbalance_ratio = static_cast<double>(*max_color) / std::max(1L, *min_color);
    
    // Choose strategy based on problem characteristics
    if(nrow < 10000) {
        // Small problems: use simple chunking
        return make_cache_blocked_sender(sched, A, r, x);
    } else if(imbalance_ratio > 3.0) {
        // Highly imbalanced colors: use interleaved approach
        return make_interleaved_sender(sched, A, r, x);
    } else {
        // Balanced colors: use cache-blocked approach
        return make_cache_blocked_sender(sched, A, r, x);
    }
}

int ComputeSYMGS_stdexec_interleaved(const SparseMatrix &A, const Vector &r, Vector &x) {
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

    auto work = make_interleaved_sender(scheduler, A, r, x);
    sync_wait(std::move(work)); // Single sync_wait with optimal cache performance
    
    return 0;
}

int ComputeSYMGS_stdexec_cache_blocked(const SparseMatrix &A, const Vector &r, Vector &x) {
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

    auto work = make_cache_blocked_sender(scheduler, A, r, x);
    sync_wait(std::move(work)); // Single sync_wait with cache blocking
    
    return 0;
}

int ComputeSYMGS_stdexec_adaptive(const SparseMatrix &A, const Vector &r, Vector &x) {
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

    auto work = make_adaptive_sender(scheduler, A, r, x);
    sync_wait(std::move(work)); // Single sync_wait with adaptive strategy selection
    
    return 0;
}