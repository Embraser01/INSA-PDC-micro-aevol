#pragma once

// PRNG
unsigned long long *gpu_counters;

typedef struct CudaMem {
    double *input;
    int *output;
} CudaMem;

CudaMem cudaMem;