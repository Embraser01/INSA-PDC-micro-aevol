class ExpManager;

/**
 * Data transfer functions
 */
void init_cuda_mem(ExpManager *exp_m);
void clean_cuda_mem();
void selection_in(ExpManager *exp_m);
void selection_out(ExpManager *exp_m);
void prom_term_in(ExpManager *exp_m, uint indiv_id);
void prom_term_out(ExpManager *exp_m);

void clean(ExpManager* exp_m);
void allocate_next_gen(int nb_indiv);
/**
 * Run all kernel for one generation -- all individuals
 */
void
run_a_step_on_GPU(ExpManager *exp_m, double w_max, double selection_pressure, bool first_gen);
