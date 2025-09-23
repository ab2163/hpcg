#
# HPCG Benchmark CMake configuration
#
cmake_minimum_required( VERSION 3.0 FATAL_ERROR )

project( hpcg
    VERSION 3.1
    LANGUAGES CXX
    DESCRIPTION "High Performance Conjugate Gradient Benchmark (HPCG) " )

#
# Options
#
option(HPCG_ENABLE_CONTIGUOUS_ARRAYS "Enable contiguous arrays for better cache pre-fetch" OFF)
option(HPCG_ENABLE_CUBIC_RADICAL_SEARCH "Enable faster search for optimal 3D process grid" OFF)
option(HPCG_ENABLE_DEBUG "Enable debug build" OFF)
option(HPCG_ENABLE_DETAILED_DEBUG "Enable detailed debug build" OFF)
option(HPCG_ENABLE_MPI "Enable MPI support" OFF)
option(HPCG_ENABLE_LONG_LONG "Enable use of 'long long' type for global indices" ON)
option(HPCG_ENABLE_OPENMP "Enable OpenMP support" OFF)
option(HPCG_ENABLE_STDEXEC "Enable stdexec optimized kernels" OFF)

add_executable( xhpcg src/main.cpp src/CG.cpp src/CG_ref.cpp src/TestCG.cpp
    src/ComputeResidual.cpp src/ExchangeHalo.cpp src/GenerateGeometry.cpp
    src/GenerateProblem.cpp src/GenerateProblem_ref.cpp src/CheckProblem.cpp
    src/OptimizeProblem.cpp src/ReadHpcgDat.cpp src/ReportResults.cpp
    src/SetupHalo.cpp src/SetupHalo_ref.cpp src/TestSymmetry.cpp
    src/TestNorms.cpp src/WriteProblem.cpp src/YAML_Doc.cpp
    src/YAML_Element.cpp src/ComputeDotProduct.cpp
    src/ComputeDotProduct_ref.cpp src/finalize.cpp src/init.cpp src/mytimer.cpp
    src/ComputeSPMV.cpp src/ComputeSPMV_ref.cpp src/ComputeSYMGS.cpp
    src/ComputeSYMGS_ref.cpp src/ComputeWAXPBY.cpp src/ComputeWAXPBY_ref.cpp
    src/ComputeMG_ref.cpp src/ComputeMG.cpp src/ComputeProlongation_ref.cpp
    src/ComputeRestriction_ref.cpp src/GenerateCoarseProblem.cpp
    src/ComputeOptimalShapeXYZ.cpp src/MixedBaseCounter.cpp
    src/CheckAspectRatio.cpp src/OutputFile.cpp
    src/ComputeSYMGS_stdexec.cpp src/ComputeSYMGS_stdexec_advanced.cpp
    src/ComputeSYMGS_stdexec_single_bulk.cpp src/ComputeSYMGS_stdexec_optimized_bulk.cpp
    src/BenchmarkSYMGS.cpp)

if (HPCG_ENABLE_CONTIGUOUS_ARRAYS)
    target_compile_definitions(xhpcg PRIVATE HPCG_CONTIGUOUS_ARRAYS)
endif ()

if (HPCG_ENABLE_CUBIC_RADICAL_SEARCH)
    target_compile_definitions(xhpcg PRIVATE HPCG_CUBIC_RADICAL_SEARCH)
endif ()

if (HPCG_ENABLE_DEBUG)
    target_compile_definitions(xhpcg PRIVATE HPCG_DEBUG)
endif ()

if (HPCG_ENABLE_DETAILED_DEBUG)
    target_compile_definitions(xhpcg PRIVATE HPCG_DETAILED_DEBUG)
endif ()

if (HPCG_ENABLE_MPI)
    set(MPI_CXX_SKIP_MPICXX ON)
    find_package(MPI REQUIRED)
    target_link_libraries(xhpcg ${MPI_CXX_LIBRARIES})
else ()
    target_compile_definitions(xhpcg PRIVATE HPCG_NO_MPI)
endif ()

if (NOT HPCG_ENABLE_LONG_LONG)
    target_compile_definitions(xhpcg PRIVATE HPCG_NO_LONG_LONG)
endif ()

if (HPCG_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(xhpcg ${OpenMP_CXX_LIBRARIES})
else ()
    target_compile_definitions(xhpcg PRIVATE HPCG_NO_OPENMP)
endif ()

if (HPCG_ENABLE_STDEXEC)
    target_compile_definitions(xhpcg PRIVATE SELECT_STDEXEC)
    # Add stdexec include directory and link libraries
    # You may need to adjust the path based on your stdexec installation
    find_package(stdexec REQUIRED)
    target_link_libraries(xhpcg stdexec::stdexec)
    
    # Enable C++20 for stdexec
    set_property(TARGET xhpcg PROPERTY CXX_STANDARD 20)
    set_property(TARGET xhpcg PROPERTY CXX_STANDARD_REQUIRED ON)
endif ()
