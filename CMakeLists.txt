cmake_minimum_required(VERSION 3.8.2)
project(pdc_mini_aevol)

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(SFMT-src-1.4)
add_definitions(-DSFMT_MEXP=607)

option(USE_CUDA "Use CUDA" ON)

# Create project library (without any CUDA files)
add_library(mini_aevol
        ExpManager.cpp
        ExpManager.h
        AeTime.cpp
        AeTime.h
        JumpingMT.cpp
        JumpingMT.h
        DnaMutator.cpp
        DnaMutator.h
        MutationEvent.cpp
        MutationEvent.cpp
        Organism.cpp
        Organism.h
        Promoter.h
        RNA.h
        Protein.h
        Stats.cpp
        Stats.h
        Gaussian.h
        Threefry.cpp
        Dna.cpp
        )

target_include_directories(mini_aevol INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Add ZLIB as dependency
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
    target_link_libraries(mini_aevol PUBLIC sfmt PUBLIC ${ZLIB_LIBRARIES})
endif (ZLIB_FOUND)

# Create the executable without CUDA
add_executable(pdc_mini_aevol main.cpp)
target_link_libraries(pdc_mini_aevol PUBLIC mini_aevol)

# Enable CUDA
if (USE_CUDA)
    enable_language("CUDA")
    set(CUDA_NVCC_FLAGS "-std=c++11 -arch=compute_60 -lineinfo")
    add_executable(pdc_mini_aevol_gpu main.cpp Algorithms.cu ThreefryGPU.cu)
    set_property(TARGET pdc_mini_aevol_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(pdc_mini_aevol_gpu PUBLIC mini_aevol)
    target_compile_definitions(pdc_mini_aevol_gpu PRIVATE USE_CUDA=1)
endif ()