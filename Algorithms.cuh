#pragma once
#include <cstdint>

// PRNG
unsigned long long *gpu_counters;

typedef struct CudaMem {
    double *fitness;
    int *nextReproducer;
    char *DNA;
    uint16_t *promoters;
    uint8_t *promoter_errors;
    uint16_t *terminators;
    uint *pos_prom_counter;
    uint *pos_term_counter;
} CudaMem;

CudaMem cudaMem;

typedef struct HostMem {
    double *fitnessArray;
    uint16_t *promoters;
    uint8_t *promoter_errors;
    uint16_t *terminators;
    uint *pos_prom_counter;
    uint *pos_term_counter;
} HostMem;

HostMem hostMem;
