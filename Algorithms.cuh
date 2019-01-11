#pragma once

// PRNG
unsigned long long *gpu_counters;

typedef struct CudaMem {
    void *input;
    void *output;
} CudaMem;

CudaMem cudaMem;