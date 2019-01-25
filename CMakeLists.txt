cmake_minimum_required(VERSION 3.8.2)
project(pdc_mini_aevol)

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(SFMT-src-1.4)
add_definitions(-DSFMT_MEXP=607)

option(USE_CUDA "Use CUDA" ON)

# Create project library (without any CUDA files)
set(SOURCE_FILES
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
        Dna.cpp)

add_library(mini_aevol ${SOURCE_FILES})
target_include_directories(mini_aevol INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})


# Create the executable without CUDA
add_executable(pdc_mini_aevol main.cpp)
target_link_libraries(pdc_mini_aevol PUBLIC mini_aevol)

# Enable CUDA
if (USE_CUDA)
    enable_language("CUDA")

    # Create the core lib with custom definitions
    add_library(mini_aevol_gpu ${SOURCE_FILES})
    target_include_directories(mini_aevol_gpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(mini_aevol_gpu PUBLIC USE_CUDA=1)

    # CUDA Files
    set(CMAKE_CUDA_FLAGS "-std=c++11 -arch=compute_50 -lineinfo")
    add_library(cuda_part STATIC Algorithms.cu Algorithms.cuh)
    set_target_properties(cuda_part PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Executable
    add_executable(pdc_mini_aevol_gpu main.cpp)
    target_link_libraries(pdc_mini_aevol_gpu PUBLIC mini_aevol_gpu cuda_part)
endif ()

# Add ZLIB as dependency
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
    target_link_libraries(mini_aevol PUBLIC sfmt PUBLIC ${ZLIB_LIBRARIES})

    if (USE_CUDA)
        target_link_libraries(mini_aevol_gpu PUBLIC sfmt PUBLIC ${ZLIB_LIBRARIES})
    endif ()
endif (ZLIB_FOUND)