#include "CG_stdexec.hpp"

int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
  const int max_iter, const double tolerance, int &niters, double &normr,  double &normr0,
  double *times, bool doPreconditioning){

  //********** DATA VARIABLES **********//

  double time_mg_0 = 0.0, time_mg_1 = 0.0, time_mg_2 = 0.0, time_mg_3 = 0.0, dummy_time = 0.0;
  double start_time = mytimer();
  double *rtz    = new double(0.0);
  double *oldrtz = new double(0.0);
  double *alpha  = new double(0.0);
  double *beta   = new double(0.0);
  double *pAp    = new double(0.0);
  normr = 0.0;
  Vector &p = data.p; //direction vector (in MPI mode ncol>=nrow)
  Vector &Ap = data.Ap;
  //for the sparse matrix at different MG depths:
  double ***A_vals = new double**[NUM_MG_LEVELS];
  local_int_t ***A_inds = new local_int_t**[NUM_MG_LEVELS];
  double ***A_diags = new double**[NUM_MG_LEVELS];
  local_int_t *A_nrows = new local_int_t[NUM_MG_LEVELS];
  char **A_nnzs = new char*[NUM_MG_LEVELS];
  unsigned char **A_colors = new unsigned char*[NUM_MG_LEVELS];
  //various values at different MG depths:
  double **r_vals = new double*[NUM_MG_LEVELS];
  double **z_vals = new double*[NUM_MG_LEVELS];
  double **Axfv_vals = new double*[NUM_MG_LEVELS - 1];
  local_int_t **f2c_vals = new local_int_t*[NUM_MG_LEVELS - 1];
  double **xcv_vals = new double*[NUM_MG_LEVELS - 1];
  double **rcv_vals = new double*[NUM_MG_LEVELS - 1];
  //objects used to initialise other data:
  const SparseMatrix **A_objs = new const SparseMatrix*[NUM_MG_LEVELS];
  Vector **r_objs= new Vector*[NUM_MG_LEVELS];
  Vector **z_objs = new Vector*[NUM_MG_LEVELS];
  //variables used by main CG algorithm:
  double * const p_vals = p.values;
  double * const x_vals = x.values;
  double * const Ap_vals = Ap.values;
  const double * const b_vals = b.values;
  //used for parallel SYMGS:
  int *color = new int;
  *color = 0;
  //used to ensure all variables within pipeline on heap
  //since CPU-GPU managed memory only works for heap vars:
  double *normr_cpy = new double(0.0);
  double *normr0_cpy = new double;

  //set object pointers to respective values
  A_objs[0] = &A;
  r_objs[0] = &data.r; //residual vector
  z_objs[0] = &data.z; //preconditioned residual vector
  for(int depth = 1; depth < NUM_MG_LEVELS; depth++){
    A_objs[depth] = A_objs[depth - 1]->Ac;
    r_objs[depth] = A_objs[depth - 1]->mgData->rc;
    z_objs[depth] = A_objs[depth - 1]->mgData->xc;
  }

  //use object pointers to set value pointers
  for(int depth = 0; depth < NUM_MG_LEVELS; depth++){
    A_vals[depth] = A_objs[depth]->matrixValues;
    A_inds[depth] = A_objs[depth]->mtxIndL;
    A_diags[depth] = A_objs[depth]->matrixDiagonal;
    A_nrows[depth] = A_objs[depth]->localNumberOfRows;
    A_nnzs[depth] = A_objs[depth]->nonzerosInRow;
    A_colors[depth]  = A_objs[depth]->colors;
    r_vals[depth] = r_objs[depth]->values;
    z_vals[depth] = z_objs[depth]->values;

    if(depth < NUM_MG_LEVELS - 1){
      Axfv_vals[depth] = A_objs[depth]->mgData->Axf->values;
      rcv_vals[depth] = A_objs[depth]->mgData->rc->values;
      f2c_vals[depth] = A_objs[depth]->mgData->f2cOperator;
      xcv_vals[depth] = A_objs[depth]->mgData->xc->values;
    }
  }
  //used in some kernels:
  local_int_t &nrow = A_nrows[0];

  //used by dot product kernel:
  double *bin_vals = new double[NUM_BINS];
  double *dot_local_result = new double(0.0);

  unsigned int num_threads = omp_get_max_threads();
  exec::static_thread_pool pool(num_threads);

#ifdef USE_GPU
  //scheduler for GPU execution
  nvexec::stream_context ctx;
  auto scheduler = ctx.get_scheduler();
#else
  //scheduler for CPU execution
  auto scheduler = pool.get_scheduler();
#endif

//********** GETTING NAMED POINTERS FOR EACH PARAMETER TO SPEED UP KERNELS **********//

double** A_vals0 = A_vals[0];
double** A_vals1 = A_vals[1];
double** A_vals2 = A_vals[2];
double** A_vals3 = A_vals[3];

local_int_t** A_inds0 = A_inds[0];
local_int_t** A_inds1 = A_inds[1];
local_int_t** A_inds2 = A_inds[2];
local_int_t** A_inds3 = A_inds[3];

double** A_diags0 = A_diags[0];
double** A_diags1 = A_diags[1];
double** A_diags2 = A_diags[2];
double** A_diags3 = A_diags[3];

local_int_t A_nrows0 = A_nrows[0];
local_int_t A_nrows1 = A_nrows[1];
local_int_t A_nrows2 = A_nrows[2];
local_int_t A_nrows3 = A_nrows[3];

char* A_nnzs0 = A_nnzs[0];
char* A_nnzs1 = A_nnzs[1];
char* A_nnzs2 = A_nnzs[2];
char* A_nnzs3 = A_nnzs[3];

unsigned char* A_colors0 = A_colors[0];
unsigned char* A_colors1 = A_colors[1];
unsigned char* A_colors2 = A_colors[2];
unsigned char* A_colors3 = A_colors[3];

double* r_vals0 = r_vals[0];
double* r_vals1 = r_vals[1];
double* r_vals2 = r_vals[2];
double* r_vals3 = r_vals[3];

double* z_vals0 = z_vals[0];
double* z_vals1 = z_vals[1];
double* z_vals2 = z_vals[2];
double* z_vals3 = z_vals[3];

double* Axfv_vals0 = Axfv_vals[0];
double* Axfv_vals1 = Axfv_vals[1];
double* Axfv_vals2 = Axfv_vals[2];

local_int_t* f2c_vals0 = f2c_vals[0];
local_int_t* f2c_vals1 = f2c_vals[1];
local_int_t* f2c_vals2 = f2c_vals[2];

double* xcv_vals0 = xcv_vals[0];
double* xcv_vals1 = xcv_vals[1];
double* xcv_vals2 = xcv_vals[2];

double* rcv_vals0 = rcv_vals[0];
double* rcv_vals1 = rcv_vals[1];
double* rcv_vals2 = rcv_vals[2];

//********** DEFINITION OF KERNELS **********//

auto dot_prod_stg1 = [=](local_int_t i, const double * const vec1_vals, const double * const vec2_vals){ 
    local_int_t minInd = i*(nrow/NUM_BINS);
    local_int_t maxInd = ((i + 1) == NUM_BINS) ? nrow : (i + 1)*(nrow/NUM_BINS);
    double bin_sum = 0.0;
    for(local_int_t j = minInd; j < maxInd; ++j){
      bin_sum = std::fma(vec1_vals[j], vec2_vals[j], bin_sum);
    }
    bin_vals[i] = bin_sum;
};

auto dot_prod_stg2 = [=](double *result){
    double result_cpy = 0.0;
    for(local_int_t i = 0; i < NUM_BINS; ++i) result_cpy += bin_vals[i];
    *result = result_cpy;
};

auto waxpby = [=](local_int_t i, double alpha, const double * const xvals, double beta, const double * const yvals, double *wvals){
  wvals[i] = alpha*xvals[i] + beta*yvals[i]; 
};

auto spmv = [=](local_int_t i, const double * const * const avals, const double * const xvals, double *yvals, 
  const local_int_t * const * const indvals, const char * const nnz){
    double sum = 0.0;
    for(int j = 0; j < nnz[i]; ++j){
      sum += avals[i][j] * xvals[indvals[i][j]];
    }
    yvals[i] = sum;
};

auto restriction = [=](local_int_t i, double *rcv_vals, const double * const r_vals, const local_int_t * const f2c_vals, 
  const double * const Axfv_vals){
  rcv_vals[i] = r_vals[f2c_vals[i]] - Axfv_vals[f2c_vals[i]];
};

auto prolongation = [=](local_int_t i, double *z_vals, const local_int_t * const f2c_vals, const double * const xcv_vals){
  z_vals[f2c_vals[i]] += xcv_vals[i];
};

auto symgs = [=](local_int_t i, const double * const * const avals, double *xvals, const double * const rvals, 
  const char * const nnz, const local_int_t * const * const indvals, const double * const * const diagvals, 
  const unsigned char * const colors){
  if(colors[i] == *color){
    const double currentDiagonal = diagvals[i][0];
    double sum = rvals[i];
    for(int j = 0; j < nnz[i]; ++j){
      local_int_t curCol = indvals[i][j];
      sum -= avals[i][j] * xvals[curCol];
    }
    sum += xvals[i]*currentDiagonal;
    xvals[i] = sum/currentDiagonal;
  }
};

//********** DEFINITION OF KERNEL "SPECIALISATIONS" **********//

auto symgs_0 = [=](local_int_t i){
  symgs(i, A_vals0, z_vals0, r_vals0, A_nnzs0, A_inds0, A_diags0, A_colors0);
};
auto symgs_1 = [=](local_int_t i){
  symgs(i, A_vals1, z_vals1, r_vals1, A_nnzs1, A_inds1, A_diags1, A_colors1);
};
auto symgs_2 = [=](local_int_t i){
  symgs(i, A_vals2, z_vals2, r_vals2, A_nnzs2, A_inds2, A_diags2, A_colors2);
};
auto symgs_3 = [=](local_int_t i){
  symgs(i, A_vals3, z_vals3, r_vals3, A_nnzs3, A_inds3, A_diags3, A_colors3);
};

auto restriction_0 = [=](local_int_t i){
  restriction(i, rcv_vals0, r_vals0, f2c_vals0, Axfv_vals0);
};
auto restriction_1 = [=](local_int_t i){
  restriction(i, rcv_vals1, r_vals1, f2c_vals1, Axfv_vals1);
};
auto restriction_2 = [=](local_int_t i){
  restriction(i, rcv_vals2, r_vals2, f2c_vals2, Axfv_vals2);
};

auto prolongation_0 = [=](local_int_t i){
  prolongation(i, z_vals0, f2c_vals0, xcv_vals0);
};
auto prolongation_1 = [=](local_int_t i){
  prolongation(i, z_vals1, f2c_vals1, xcv_vals1);
};
auto prolongation_2 = [=](local_int_t i){
  prolongation(i, z_vals2, f2c_vals2, xcv_vals2);
};

auto spmv_mg0 = [=](local_int_t i){
  spmv(i, A_vals0, z_vals0, Axfv_vals0, A_inds0, A_nnzs0);
};
auto spmv_mg1 = [=](local_int_t i){
  spmv(i, A_vals1, z_vals1, Axfv_vals1, A_inds1, A_nnzs1);
};
auto spmv_mg2 = [=](local_int_t i){
  spmv(i, A_vals2, z_vals2, Axfv_vals2, A_inds2, A_nnzs2);
};
auto spmv_Ap = [=](local_int_t i){
  spmv(i, A_vals0, p_vals, Ap_vals, A_inds0, A_nnzs0);
};

auto waxpby_peqx = [=](local_int_t i){
  waxpby(i, 1, x_vals, 0, x_vals, p_vals);
};
auto waxpby_reqbmAp = [=](local_int_t i){
  waxpby(i, 1, b_vals, -1, Ap_vals, r_vals0);
};
auto waxpby_peqz = [=](local_int_t i){
  waxpby(i, 1, z_vals0, 0, z_vals0, p_vals);
};
auto waxpby_xeqxpap = [=](local_int_t i){
  waxpby(i, 1, x_vals, *alpha, p_vals, x_vals);
};
auto waxpby_reqrmaAp = [=](local_int_t i){
  waxpby(i, 1, r_vals0, -*alpha, Ap_vals, r_vals0);
};
auto waxpby_peqbppz = [=](local_int_t i){
  waxpby(i, 1, z_vals0, *beta, p_vals, p_vals);
};
auto zerovector_0 = [=](local_int_t i){
  waxpby(i, 0, z_vals0, 0, z_vals0, z_vals0);
};
auto zerovector_1 = [=](local_int_t i){
  waxpby(i, 0, z_vals1, 0, z_vals1, z_vals1);
};
auto zerovector_2 = [=](local_int_t i){
  waxpby(i, 0, z_vals2, 0, z_vals2, z_vals2);
};
auto zerovector_3 = [=](local_int_t i){
  waxpby(i, 0, z_vals3, 0, z_vals3, z_vals3);
};

auto dot_prod_rz_stg1 = [=](local_int_t i){
  dot_prod_stg1(i, r_vals0, z_vals0);
};
auto dot_prod_rz_stg2 = [=](){
  dot_prod_stg2(rtz);
};
auto dot_prod_pAp_stg1 = [=](local_int_t i){
  dot_prod_stg1(i, p_vals, Ap_vals);
};
auto dot_prod_pAp_stg2 = [=](){
  dot_prod_stg2(pAp);
};
auto dot_prod_rr_stg1 = [=](local_int_t i){
  dot_prod_stg1(i, r_vals0, r_vals0);
};
auto dot_prod_rr_stg2 = [=](){
  dot_prod_stg2(normr_cpy);
};

//********** START OF RUNNING PROGRAM *********//

  if (!doPreconditioning && A.geom->rank == 0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
#endif

  sender auto pre_loop_work = schedule(scheduler)
  | bulk(par_unseq, nrow, waxpby_peqx)
  | bulk(par_unseq, nrow, spmv_Ap)
  | bulk(par_unseq, nrow, waxpby_reqbmAp)
  | bulk(par_unseq, NUM_BINS, dot_prod_rr_stg1)
  | then(dot_prod_rr_stg2)
  | then([=](){
    *normr_cpy = sqrt(*normr_cpy);
    *normr0_cpy = *normr_cpy; //record initial residual for convergence testing
  });
  sync_wait(std::move(pre_loop_work));

#ifdef HPCG_DEBUG
    if (A.geom->rank == 0) HPCG_fout << "Initial Residual = "<< *normr_cpy << std::endl;
#endif
  
  int k = 1;

  //mg process first stage
  sender auto mg_stg0 = schedule(scheduler)
    | bulk(par_unseq, A_nrows0, zerovector_0)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | bulk(par_unseq, A_nrows0, spmv_mg0)
    | bulk(par_unseq, A_nrows1, restriction_0);

  sender auto mg_stg1 = schedule(scheduler)
    | bulk(par_unseq, A_nrows1, zerovector_1)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | bulk(par_unseq, A_nrows1, spmv_mg1)
    | bulk(par_unseq, A_nrows2, restriction_1);

  sender auto mg_stg2 = schedule(scheduler)
    | bulk(par_unseq, A_nrows2, zerovector_2)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | bulk(par_unseq, A_nrows2, spmv_mg2)
    | bulk(par_unseq, A_nrows3, restriction_2);

  sender auto mg_stg3 = schedule(scheduler)
    | bulk(par_unseq, A_nrows3, zerovector_3)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows3, symgs_3);

  sender auto mg_stg4 = schedule(scheduler)
    | bulk(par_unseq, A_nrows3, prolongation_2)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows2, symgs_2);
  
  sender auto mg_stg5 = schedule(scheduler)
    | bulk(par_unseq, A_nrows2, prolongation_1)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows1, symgs_1);
  
  sender auto mg_stg6 = schedule(scheduler)
    | bulk(par_unseq, A_nrows1, prolongation_0)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ *color = 0; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0)
    | then([=](){ (*color)++; })
    | bulk(par_unseq, A_nrows0, symgs_0);

  dummy_time = mytimer();
  sync_wait(mg_stg0);
  time_mg_0 += (mytimer() - dummy_time);

  dummy_time = mytimer();
  sync_wait(mg_stg1);
  time_mg_1 += (mytimer() - dummy_time);

  dummy_time = mytimer();
  sync_wait(mg_stg2);
  time_mg_2 += (mytimer() - dummy_time);

  dummy_time = mytimer();
  sync_wait(mg_stg3);
  time_mg_3 += (mytimer() - dummy_time);

  dummy_time = mytimer();
  sync_wait(mg_stg4);
  time_mg_2 += (mytimer() - dummy_time);

  dummy_time = mytimer();
  sync_wait(mg_stg5);
  time_mg_1 += (mytimer() - dummy_time);

  dummy_time = mytimer();
  sync_wait(mg_stg6);
  time_mg_0 += (mytimer() - dummy_time);
  
  sender auto rest_of_first_loop = schedule(scheduler)
    | bulk(par_unseq, nrow, waxpby_peqz)
    | bulk(par_unseq, NUM_BINS, dot_prod_rz_stg1)
    | then(dot_prod_rz_stg2)
    | bulk(par_unseq, nrow, spmv_Ap)
    | bulk(par_unseq, NUM_BINS, dot_prod_pAp_stg1)
    | then(dot_prod_pAp_stg2)
    | then([=](){ *alpha = *rtz/(*pAp); })
    | bulk(par_unseq, nrow, waxpby_xeqxpap)
    | bulk(par_unseq, nrow, waxpby_reqrmaAp)
    | bulk(par_unseq, NUM_BINS, dot_prod_rr_stg1)
    | then(dot_prod_rr_stg2)
    | then([=](){ *normr_cpy = sqrt(*normr_cpy); });
  sync_wait(std::move(rest_of_first_loop));

#ifdef HPCG_DEBUG
    if(A.geom->rank == 0 && (1 % print_freq == 0 || 1 == max_iter))
      HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< *normr_cpy/(*normr0_cpy) << std::endl;
#endif
    niters = 1;

  sender auto rest_of_loop = schedule(scheduler)
    | then([=](){ *oldrtz = *rtz; })
    | bulk(par_unseq, NUM_BINS, dot_prod_rz_stg1)
    | then(dot_prod_rz_stg2)
    | then([=](){ *beta = *rtz/(*oldrtz); })
    | bulk(par_unseq, nrow, waxpby_peqbppz)
    | bulk(par_unseq, nrow, spmv_Ap)
    | bulk(par_unseq, NUM_BINS, dot_prod_pAp_stg1)
    | then(dot_prod_pAp_stg2)
    | then([=](){ *alpha = *rtz/(*pAp); })
    | bulk(par_unseq, nrow, waxpby_xeqxpap)
    | bulk(par_unseq, nrow, waxpby_reqrmaAp)
    | bulk(par_unseq, NUM_BINS, dot_prod_rr_stg1)
    | then(dot_prod_rr_stg2)
    | then([=](){ *normr_cpy = sqrt(*normr_cpy); });

  //start iterations
  for(k = 2; k <= max_iter && *normr_cpy/(*normr0_cpy) > tolerance; k++){

    dummy_time = mytimer();
    sync_wait(mg_stg0);
    time_mg_0 += (mytimer() - dummy_time);

    dummy_time = mytimer();
    sync_wait(mg_stg1);
    time_mg_1 += (mytimer() - dummy_time);

    dummy_time = mytimer();
    sync_wait(mg_stg2);
    time_mg_2 += (mytimer() - dummy_time);

    dummy_time = mytimer();
    sync_wait(mg_stg3);
    time_mg_3 += (mytimer() - dummy_time);

    dummy_time = mytimer();
    sync_wait(mg_stg4);
    time_mg_2 += (mytimer() - dummy_time);

    dummy_time = mytimer();
    sync_wait(mg_stg5);
    time_mg_1 += (mytimer() - dummy_time);
    
    dummy_time = mytimer();
    sync_wait(mg_stg6);
    time_mg_0 += (mytimer() - dummy_time);

    sync_wait(rest_of_loop);

#ifdef HPCG_DEBUG
    if(A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
      HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< *normr_cpy/(*normr0_cpy) << std::endl;
#endif
    niters = k;
  }

  std::cout << "Time for MG level 0: " << time_mg_0 << ".\n";
  std::cout << "Time for MG level 1: " << time_mg_1 << ".\n";
  std::cout << "Time for MG level 2: " << time_mg_2 << ".\n";
  std::cout << "Time for MG level 3: " << time_mg_3 << ".\n";

  times[0] = mytimer() - start_time; //record total time elapsed
  normr = *normr_cpy;
  normr0 = *normr0_cpy;
  delete color;
  delete alpha;
  delete beta;
  delete rtz;
  delete oldrtz;
  delete pAp;
  delete normr_cpy;
  delete normr0_cpy;
  delete A_vals;
  delete A_inds;
  delete A_diags;
  delete A_nrows;
  delete A_nnzs;
  delete A_colors;
  delete r_vals;
  delete z_vals;
  delete Axfv_vals;
  delete f2c_vals;
  delete xcv_vals;
  delete rcv_vals;
  delete A_objs;
  delete r_objs;
  delete z_objs;
  delete bin_vals;
  delete dot_local_result;
  return 0;
}