#include "Algorithms.h"
#include "Algorithms.cuh"

#include "ExpManager.h"
#include "ThreefryGPU.h"
#include "GPUDna.cuh"

#include <cmath>
#include <map>
#include <algorithm>
#include <sys/stat.h>
#include <err.h>
#include<chrono>
#include <iostream>

#include "AeTime.h"
#include "Threefry.h"
#include "DnaMutator.h"
#include "Stats.h"
#include "ExpManager.h"
#include "Algorithms.h"
#include "AeTime.h"
#include "Promoter.h"
#include "RNA.h"
#include "Protein.h"
#include "Organism.h"
#include "Gaussian.h"

using namespace std::chrono;


#include <cstdint>
#include <stdio.h>
#include <unistd.h>

#include <iostream>

#include<cuda.h>
#include<cuda_profiler_api.h>

using namespace std;

#define DEBUG 1

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


constexpr int32_t PROMOTER_ARRAY_SIZE = 10000;

/**
 * Function to transfer data from CPU to GPU
 *
 * @param exp_m
 * @param first_gen
 */
void transfer_in(ExpManager *exp_m, bool first_gen) {
    exp_m->rng_->initDevice();
    checkCuda(cudaMalloc((void **) &gpu_counters,
                         exp_m->rng_->counters().size() *
                         sizeof(unsigned long long)));

    checkCuda(cudaMemcpy(gpu_counters, exp_m->rng_->counters().data(),
                         exp_m->rng_->counters().size() *
                         sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // TO COMPLETE
}


__device__ int32_t

Threefry::Device::roulette_random(double *probs, int32_t nb_elts) {
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


__constant__ double cof[6] = {76.18009172947146,
                              -86.50532032941677,
                              24.01409824083091,
                              -1.231739572450155,
                              0.1208650973866179e-2,
                              -0.5395239384953e-5};


// Returns the value ln[gamma(X)] for X.
// The gamma function is defined by the integral  gamma(z) = int(0, +inf, t^(z-1).e^(-t)dt).
// When the argument z is an integer, the gamma function is just the familiar factorial
// function, but offset by one, n! = gamma(n + 1).
__device__ static double gammln(double X) {
    double x, y, tmp, ser;

    y = x = X;
    tmp = x + 5.5;
    tmp -= (x + 0.5) * log(tmp);
    ser = 1.000000000190015;

    for (int8_t j = 0; j <= 5; j++) {
        ser += cof[j] / ++y;
    }

    return -tmp + log(2.5066282746310005 * ser / x);
}


__device__ int32_t Threefry::Device::binomial_random(int32_t nb_drawings, double prob) {
    int32_t nb_success;

    // The binomial distribution is invariant under changing
    // ProbSuccess to 1-ProbSuccess, if we also change the answer to
    // NbTrials minus itself; we ll remember to do this below.
    double p;
    if (prob <= 0.5) p = prob;
    else p = 1.0 - prob;

    // mean of the deviate to be produced
    double mean = nb_drawings * p;


    if (nb_drawings < 25)
        // Use the direct method while NbTrials is not too large.
        // This can require up to 25 calls to the uniform random.
    {
        nb_success = 0;
        for (int32_t j = 1 ; j <= nb_drawings ; j++)
        {
            if (randomDouble() < p) nb_success++;
        }
    }
    else if (mean < 1.0)
        // If fewer than one event is expected out of 25 or more trials,
        // then the distribution is quite accurately Poisson. Use direct Poisson method.
    {
        double g = exp(-mean);
        double t = 1.0;
        int32_t j;
        for (j = 0; j <= nb_drawings ; j++)
        {
            t = t * randomDouble();
            if (t < g) break;
        }

        if (j <= nb_drawings) nb_success = j;
        else nb_success = nb_drawings;
    }

    else
        // Use the rejection method.
    {
        double en     = nb_drawings;
        double oldg   = gammln(en + 1.0);
        double pc     = 1.0 - p;
        double plog   = log(p);
        double pclog  = log(pc);

        // rejection method with a Lorentzian comparison function.
        double sq = sqrt(2.0 * mean * pc);
        double angle, y, em, t;
        do
        {
            do
            {
                angle = M_PI * randomDouble();
                y = tan(angle);
                em = sq*y + mean;
            } while (em < 0.0 || em >= (en + 1.0)); // Reject.

            em = floor(em); // Trick for integer-valued distribution.
            t = 1.2 * sq * (1.0 + y*y)
                * exp(oldg - gammln(em + 1.0) - gammln(en - em + 1.0) + em * plog + (en - em) * pclog);

        } while (randomDouble() > t); // Reject. This happens about 1.5 times per deviate, on average.

        nb_success = (int32_t) rint(em);
    }


    // Undo the symmetry transformation.
    if (p != prob) nb_success = nb_drawings - nb_success;

    return nb_success;
}

__device__ static int mod(int a, int b) {

    assert(b > 0);

    while (a < 0) a += b;
    while (a >= b) a -= b;

    return a;
}

/**
 * Run a step on the GPU
 */
void run_a_step_on_GPU(ExpManager *exp_m, double w_max, double selection_pressure, bool first_gen) {

    // Running the simulation process for each organism
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            exp_m->selection(indiv_id);
        }
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto duration_selection = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        t1 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            exp_m->do_mutation(indiv_id);
        }
        t2 = high_resolution_clock::now();
        auto duration_mutation = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        t1 = high_resolution_clock::now();
        for (int indiv_id = 0; indiv_id < exp_m->nb_indivs_; indiv_id++) {
            if (dna_mutator_array_[indiv_id]->hasMutate()) {
                exp_m->opt_prom_compute_RNA(indiv_id);
            }
        }
        t2 = high_resolution_clock::now();
        auto duration_start_stop_RNA = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

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

        //transfer_out(this);


        std::cout << "LOG," << duration_selection << "," << duration_mutation << "," << duration_start_stop_RNA
                  << "," << duration_start_protein << "," << duration_compute_protein << ","
                  << duration_translate_protein
                  << "," << duration_compute_phenotype << "," << duration_compute_phenotype << ","
                  << duration_compute_fitness << std::endl;
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


/**
 * Reallocate some data structures if needed
 * @param nb_indiv
 */
void allocate_next_gen(int nb_indiv) {

}

/**
PRNG usage:
 * For selection
        Threefry::Device rng(gpu_counters,indiv_id,Threefry::Phase::REPROD,nb_indiv);
        int found_org = rng.roulette_random(probs, NEIGHBORHOOD_SIZE);
 * For mutation:
      Threefry::Device rng(gpu_counters,indiv_id,Threefry::Phase::MUTATION,nb_indivs);
      rng.binomial_random(prev_gen_size, mutation_r);
      rng.random( number );
 **/