#pragma once

#include "../../../../../../usr/lib/gcc/x86_64-linux-gnu/6/include/stdint-gcc.h"

// PRNG
unsigned long long *gpu_counters;

typedef struct CudaMem {
    double *fitness;
    int *nextReproducer;
    char *DNA;
    uint16_t *promoters;
    uint8_t *promoter_errors;
    uint16_t *terminators;
    uint16_t *pos_prom_counter;
    uint16_t *pos_term_counter;
} CudaMem;

CudaMem cudaMem;

typedef struct HostMem {
    double *fitnessArray;
    uint16_t *promoters;
    uint8_t *promoter_errors;
    uint16_t *terminators;
    uint16_t *pos_prom_counter;
    uint16_t *pos_term_counter;
} HostMem;

HostMem hostMem;