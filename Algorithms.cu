#include "Algorithms.h"
#include "Algorithms.cuh"

#include <sys/stat.h>
#include <chrono>
#include <iostream>
#include <map>
// #include <stdio.h>
// #include <unistd.h>

#include <cuda.h>
#include <cuda_profiler_api.h>

#include "ExpManager.h"
#include "ThreefryGPU.h"
#include "GPUDna.cuh"
#include "AeTime.h"

using namespace std;
using namespace std::chrono;

#define DEBUG 1

#define TILE_WIDTH 16
#define STD_BLOCK_SIZE 1024
constexpr int DNA_PROM_INNER_BLOCK = STD_BLOCK_SIZE - PROM_SIZE + 1;
constexpr int DNA_TERM_INNER_BLOCK = STD_BLOCK_SIZE - 10;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}


void init_cuda_mem(ExpManager *exp_m) {
    // Random numbers generator
    exp_m->rng_->initDevice();
    checkCuda(cudaMalloc((void **) &gpu_counters,
                         exp_m->rng_->counters().size() *
                         sizeof(unsigned long long)));
    checkCuda(cudaMemcpy(gpu_counters, exp_m->rng_->counters().data(),
                         exp_m->rng_->counters().size() *
                         sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Selection
    hostMem.fitnessArray= new double[exp_m->nb_indivs_];
    checkCuda(cudaMalloc((void **) &cudaMem.fitness, exp_m->nb_indivs_ * sizeof(double)));
    checkCuda(cudaMalloc((void **) &cudaMem.nextReproducer, exp_m->nb_indivs_ * sizeof(int)));

    // DNA
    checkCuda(cudaMalloc((void **) &cudaMem.DNA, exp_m->nb_indivs_ * exp_m->genome_size * sizeof(char)));

    // Promoters and terminators
    checkCuda(cudaMalloc((void **) &cudaMem.promoters, exp_m->nb_indivs_ * exp_m->genome_size * sizeof(uint16_t)));
    checkCuda(cudaMalloc((void **) &cudaMem.promoter_errors, exp_m->nb_indivs_ * exp_m->genome_size * sizeof(uint8_t)));
    checkCuda(cudaMalloc((void **) &cudaMem.terminators, exp_m->nb_indivs_ * exp_m->genome_size * sizeof(uint16_t)));
    checkCuda(cudaMalloc((void **) &cudaMem.pos_prom_counter, exp_m->nb_indivs_ * sizeof(uint)));
    checkCuda(cudaMalloc((void **) &cudaMem.pos_term_counter, exp_m->nb_indivs_ * sizeof(uint)));
    hostMem.promoters = new uint16_t[exp_m->nb_indivs_ * exp_m->genome_size];
    hostMem.promoter_errors = new uint8_t[exp_m->nb_indivs_ * exp_m->genome_size];
    hostMem.terminators = new uint16_t[exp_m->nb_indivs_ * exp_m->genome_size];
    hostMem.pos_prom_counter = new uint[exp_m->nb_indivs_];
    hostMem.pos_term_counter = new uint[exp_m->nb_indivs_];
}


void selection_in(ExpManager *exp_m) {
    for (int i = 0; i < exp_m->nb_indivs_; ++i) {
        hostMem.fitnessArray[i] = exp_m->prev_internal_organisms_[i]->fitness;
    }
    checkCuda(cudaMemcpy(cudaMem.fitness, hostMem.fitnessArray,
                         exp_m->nb_indivs_ * sizeof(double), cudaMemcpyHostToDevice));
}

void selection_out(ExpManager *exp_m) {
    checkCuda(cudaMemcpy(exp_m->next_generation_reproducer_, cudaMem.nextReproducer,
              exp_m->nb_indivs_ * sizeof(int), cudaMemcpyDeviceToHost));
}

void prom_term_in(ExpManager *exp_m, uint indiv_id) {
    if(indiv_id == 0) {
        cudaMemset(cudaMem.pos_prom_counter, 0, exp_m->nb_indivs_ * sizeof(uint));
        cudaMemset(cudaMem.pos_term_counter, 0, exp_m->nb_indivs_ * sizeof(uint));
    }

    checkCuda(cudaMemcpy(&cudaMem.DNA[indiv_id * exp_m->genome_size],
                         exp_m->internal_organisms_[indiv_id]->dna_->seq_.data(),
                         exp_m->genome_size * sizeof(char), cudaMemcpyHostToDevice));
}

void prom_term_out(ExpManager *exp_m) {
    checkCuda(cudaMemcpy(hostMem.pos_prom_counter, cudaMem.pos_prom_counter,
                         exp_m->nb_indivs_ * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(hostMem.pos_term_counter, cudaMem.pos_term_counter,
                         exp_m->nb_indivs_ * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    for(uint indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
        if (!exp_m->dna_mutator_array_[indiv_id]->hasMutate()) continue;
        checkCuda(cudaMemcpy(&hostMem.promoters[indiv_id * exp_m->genome_size],
                             &cudaMem.promoters[indiv_id * exp_m->genome_size],
                             hostMem.pos_prom_counter[indiv_id] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(&hostMem.promoter_errors[indiv_id * exp_m->genome_size],
                             &cudaMem.promoter_errors[indiv_id * exp_m->genome_size],
                             hostMem.pos_prom_counter[indiv_id] * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(&hostMem.terminators[indiv_id * exp_m->genome_size],
                             &cudaMem.terminators[indiv_id * exp_m->genome_size],
                             hostMem.pos_term_counter[indiv_id] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    for(uint indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
        if (!exp_m->dna_mutator_array_[indiv_id]->hasMutate()) continue;

        // Found promoters
        map<uint16_t, uint8_t> temp_prom;
        for(uint i = 0; i < hostMem.pos_prom_counter[indiv_id]; i++) {
            temp_prom.insert(pair<int, int>(hostMem.promoters[indiv_id * exp_m->genome_size + i],
                                            hostMem.promoter_errors[indiv_id * exp_m->genome_size + i]));
        }
        shared_ptr<Organism> &currentOrganism = exp_m->internal_organisms_[indiv_id];
        uint prom_counter = 0;
        for(const pair<uint16_t, uint8_t>& prom: temp_prom) {
            Promoter *nprom = new Promoter((int)prom.first, (int)prom.second);
            currentOrganism->promoters[prom_counter] = nprom;
            currentOrganism->count_prom++;
            prom_counter++;
        }

        // Found terminators
        for(uint i = 0; i < hostMem.pos_term_counter[indiv_id]; i++) {
            currentOrganism->terminators.insert((int)hostMem.terminators[indiv_id * exp_m->genome_size + i]);
        }
    }
}

void clean_cuda_mem() {
    // Device
    cudaFree(gpu_counters);
    cudaFree(cudaMem.fitness);
    cudaFree(cudaMem.nextReproducer);
    cudaFree(cudaMem.DNA);
    cudaFree(cudaMem.promoters);
    cudaFree(cudaMem.terminators);
    cudaFree(cudaMem.pos_prom_counter);
    cudaFree(cudaMem.pos_term_counter);

    // Host
    delete[] hostMem.fitnessArray;
    delete[] hostMem.promoters;
    delete[] hostMem.promoter_errors;
    delete[] hostMem.terminators;
    delete[] hostMem.pos_prom_counter;
    delete[] hostMem.pos_term_counter;
}


__device__ int32_t Threefry::Device::roulette_random(double *probs, int32_t nb_elts) {
    double pick_one = 0.0;

    while (pick_one == 0.0) {
        pick_one = randomDouble();
    }

    int32_t
    found_org = 0;

    pick_one -= probs[0];
    while (pick_one > 0) {
        assert(found_org < nb_elts - 1);

        pick_one -= probs[++found_org];
    }
    return found_org;
}

__device__ static int mod(int a, int b) {
    assert(b > 0);
    while (a < 0) a += b;
    while (a >= b) a -= b;
    return a;
}
__device__ static uint mod(uint a, uint b) {
    while (a >= b) a -= b;
    return a;
}


__global__ void selection_gpu_kernel(const double* fitnessArr, int* nextReproducers, uint grid_height, uint grid_width,
        unsigned long long* gpu_counters) {
    const uint8_t i_t = threadIdx.y;
    const uint8_t j_t = threadIdx.x;
    const int i_abs = blockIdx.y * (TILE_WIDTH - 2) + i_t - 1;
    const int j_abs = blockIdx.x * (TILE_WIDTH - 2) + j_t - 1;
    const uint i = mod((uint)(i_abs + grid_height), grid_height);
    const uint j = mod((uint)(j_abs + grid_width), grid_width);
    const uint indiv_id = i * grid_width + j;

    // Preload the data
    __shared__ double preload[TILE_WIDTH][TILE_WIDTH];
    preload[i_t][j_t] = fitnessArr[indiv_id];
    __syncthreads();

    if (i_t <= 0 || j_t <= 0 || i_t >= TILE_WIDTH - 1 || j_t >= TILE_WIDTH - 1
        || i_abs >= grid_height || j_abs >= grid_width) return;

    // Calculate value
    double sumLocalFit = 0.0;
    double probs[9];
    for (int8_t o_i = -1; o_i <= 1; o_i++) {
        for (int8_t o_j = -1; o_j <= 1; o_j++) {
            probs[3 * (o_i + 1) + o_j + 1] = preload[i_t + o_i][j_t + o_j];
            sumLocalFit += probs[3 * (o_i + 1) + o_j + 1];
        }
    }

    for (uint8_t k = 0; k < 9; k++) {
        probs[k] /= sumLocalFit;
    }

    Threefry::Device rng(gpu_counters, indiv_id, Threefry::Phase::REPROD, grid_width * grid_height);
    int found_org = rng.roulette_random(probs, 9);

    int i_offset = (found_org / 3) - 1;
    int j_offset = mod(found_org, 3) - 1;

    nextReproducers[indiv_id] = ((i + i_offset + grid_height) % grid_height) * grid_width +
                                ((j + j_offset + grid_width) % grid_width);
}

void selection_gpu(ExpManager *exp_m) {
    dim3 grid(ceil(exp_m->grid_width_  / (float)(TILE_WIDTH - 2)),
              ceil(exp_m->grid_height_ / (float)(TILE_WIDTH - 2)), 1);
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);

    selection_gpu_kernel <<< grid, block >>>
        (cudaMem.fitness, cudaMem.nextReproducer, exp_m->grid_height_, exp_m->grid_width_, gpu_counters);

    checkCuda(cudaDeviceSynchronize());
}


__global__ void search_promoters_gpu_kernel(
        char *DNA, uint16_t *promoters, uint8_t *promoter_errors, uint *pos_prom_counter,
        int genome_size, int indiv_id) {
    const uint8_t i_t = threadIdx.x;
    const uint16_t i = blockIdx.x * DNA_PROM_INNER_BLOCK + i_t;

    __shared__ char preload[STD_BLOCK_SIZE];
    preload[i_t] = DNA[(i < genome_size) ? (indiv_id * genome_size + i) : ((indiv_id - 1) * genome_size + i)];
    __syncthreads();

    if(i < genome_size && i_t >= DNA_PROM_INNER_BLOCK) return;

    uint8_t error = 0;
    for(uint k = 0; k < PROM_SIZE; k++) {
        error += (PROM_SEQ[k] == preload[i_t + k]) ? 0 : 1;
    }

    if(error > 4) return;

    uint pos_to_write = atomicAdd(&pos_prom_counter[indiv_id], 1);
    promoters[indiv_id * genome_size + pos_to_write] = i;
    promoter_errors[indiv_id * genome_size + pos_to_write] = error;
}

void search_promoters_gpu(ExpManager *exp_m, int indiv_id) {
    int grid = ceil(exp_m->genome_size / DNA_PROM_INNER_BLOCK);
    int block = STD_BLOCK_SIZE;
    search_promoters_gpu_kernel <<< grid, block >>>
        (cudaMem.DNA, cudaMem.promoters, cudaMem.promoter_errors, cudaMem.pos_prom_counter,
                exp_m->genome_size, indiv_id);
}


__global__ void search_terminators_gpu_kernel(char *DNA, uint16_t *terminators, uint *pos_term_counter,
                                              int genome_size, int indiv_id) {
    const uint8_t i_t = threadIdx.x;
    const uint16_t i = blockIdx.x * DNA_TERM_INNER_BLOCK + i_t;

    __shared__ char preload[STD_BLOCK_SIZE];
    preload[i_t] = DNA[(i < genome_size) ? (indiv_id * genome_size + i) : ((indiv_id - 1) * genome_size + i)];
    __syncthreads();

    if(i < genome_size && i_t >= DNA_TERM_INNER_BLOCK) return;

    uint8_t error = 0;
    for(uint k = 0; k < 4; k++) {
        error += (preload[i_t + k] == preload[i_t - k + 10]) ? 0 : 1;
    }

    if(error > 0) return;

    uint pos_to_write = atomicAdd(&pos_term_counter[indiv_id], 1);
    terminators[indiv_id * genome_size + pos_to_write] = i;
}

void search_terminators_gpu(ExpManager *exp_m, int indiv_id) {
    int grid = ceil(exp_m->genome_size / DNA_TERM_INNER_BLOCK);
    int block = STD_BLOCK_SIZE;
    search_terminators_gpu_kernel <<< grid, block >>>
        (cudaMem.DNA, cudaMem.terminators, cudaMem.pos_term_counter, exp_m->genome_size, indiv_id);
}


/**
 * Run a step on the GPU
 */
void run_a_step_on_GPU(ExpManager *exp_m, double w_max, double selection_pressure, bool first_gen) {

    // Running the simulation process for each organism
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        selection_in(exp_m);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        selection_gpu(exp_m);
        high_resolution_clock::time_point t3 = high_resolution_clock::now();
        selection_out(exp_m);
        high_resolution_clock::time_point t4 = high_resolution_clock::now();
        auto duration_selection = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count();
        auto duration_selection_calc = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        t1 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            exp_m->do_mutation(indiv_id);
        }
        t2 = high_resolution_clock::now();
        auto duration_mutation = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        t1 = high_resolution_clock::now();
        for (uint indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            if (!exp_m->dna_mutator_array_[indiv_id]->hasMutate()) continue;
            //prom_term_in(exp_m, indiv_id);
            //search_promoters_gpu(exp_m, indiv_id);
            //search_terminators_gpu(exp_m, indiv_id);
        }
        cudaDeviceSynchronize();
        //prom_term_out(exp_m);
        t2 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            if (!exp_m->dna_mutator_array_[indiv_id]->hasMutate()) continue;
            exp_m->opt_prom_compute_RNA(indiv_id);
            exp_m->compute_RNA(indiv_id);
        }
        t3 = high_resolution_clock::now();
        auto duration_start_stop_RNA = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto duration_compute_RNA = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        t1 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            if (exp_m->dna_mutator_array_[indiv_id]->hasMutate()) {
                exp_m->start_protein(indiv_id);
            }
        }
        t2 = high_resolution_clock::now();
        auto duration_start_protein = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();


        t1 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            if (exp_m->dna_mutator_array_[indiv_id]->hasMutate()) {
                exp_m->compute_protein(indiv_id);
            }
        }
        t2 = high_resolution_clock::now();
        auto duration_compute_protein = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();


        t1 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            if (exp_m->dna_mutator_array_[indiv_id]->hasMutate()) {
                exp_m->translate_protein(indiv_id, w_max);
            }
        }
        t2 = high_resolution_clock::now();
        auto duration_translate_protein = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();


        t1 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            if (exp_m->dna_mutator_array_[indiv_id]->hasMutate()) {
                exp_m->compute_phenotype(indiv_id);
            }
        }
        t2 = high_resolution_clock::now();
        auto duration_compute_phenotype = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();


        t1 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            if (exp_m->dna_mutator_array_[indiv_id]->hasMutate()) {
                exp_m->compute_fitness(indiv_id, selection_pressure);
            }
        }
        t2 = high_resolution_clock::now();
        auto duration_compute_fitness = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();


        std::cout << "LOG," << duration_selection  << "(" << duration_selection_calc << "),"
                  << duration_mutation << "," << duration_start_stop_RNA << "," << duration_compute_RNA
                  << "," << duration_start_protein << "," << duration_compute_protein << ","
                  << duration_translate_protein
                  << "," << duration_compute_phenotype << "," << duration_compute_phenotype << ","
                  << duration_compute_fitness << std::endl;
//        cout << "SEARCH," << duration_start_stop_RNA << endl;

    }
    for (int indiv_id = 1; indiv_id < exp_m->nb_indivs_; indiv_id++) {
        exp_m->prev_internal_organisms_[indiv_id] = exp_m->internal_organisms_[indiv_id];
        exp_m->internal_organisms_[indiv_id] = nullptr;
    }

    // Search for the best
    double best_fitness = exp_m->prev_internal_organisms_[0]->fitness;
    int idx_best = 0;
    for (int indiv_id = 1; indiv_id < exp_m->nb_indivs_; indiv_id++) {
        if (exp_m->prev_internal_organisms_[indiv_id]->fitness > best_fitness) {
            idx_best = indiv_id;
            best_fitness = exp_m->prev_internal_organisms_[indiv_id]->fitness;
        }
    }
    exp_m->best_indiv = exp_m->prev_internal_organisms_[idx_best];


    // Stats
    if (first_gen) {
        exp_m->stats_best = new Stats(exp_m, AeTime::time(), true);
        exp_m->stats_mean = new Stats(exp_m, AeTime::time(), false);
    } else {
        exp_m->stats_best->reinit(AeTime::time());
        exp_m->stats_mean->reinit(AeTime::time());
    }

    std::vector<int> already_seen;
    for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
        if (std::find(already_seen.begin(), already_seen.end(), indiv_id) == already_seen.end()) {
            exp_m->prev_internal_organisms_[indiv_id]->reset_stats();

            for (int i = 0; i < exp_m->prev_internal_organisms_[indiv_id]->rna_count_; i++) {
                if (exp_m->prev_internal_organisms_[indiv_id]->rnas[i] != nullptr) {
                    if (exp_m->prev_internal_organisms_[indiv_id]->rnas[i]->is_coding_)
                        exp_m->prev_internal_organisms_[indiv_id]->nb_coding_RNAs++;
                    else
                        exp_m->prev_internal_organisms_[indiv_id]->nb_non_coding_RNAs++;
                }
            }

            for (int i = 0; i < exp_m->prev_internal_organisms_[indiv_id]->protein_count_; i++) {
                if (exp_m->prev_internal_organisms_[indiv_id]->rnas[i] != nullptr) {
                    if (exp_m->prev_internal_organisms_[indiv_id]->proteins[i]->is_functional) {
                        exp_m->prev_internal_organisms_[indiv_id]->nb_func_genes++;
                    } else {
                        exp_m->prev_internal_organisms_[indiv_id]->nb_non_func_genes++;
                    }
                    if (exp_m->prev_internal_organisms_[indiv_id]->proteins[i]->h > 0) {
                        exp_m->prev_internal_organisms_[indiv_id]->nb_genes_activ++;
                    } else {
                        exp_m->prev_internal_organisms_[indiv_id]->nb_genes_inhib++;
                    }
                }
            }
        }
    }

    exp_m->stats_best->write_best();
    exp_m->stats_mean->write_average();
}
