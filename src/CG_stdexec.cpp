#include "CG_stdexec.hpp"

int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
  const int max_iter, const double tolerance, int &niters, double &normr,  double &normr0,
  double *times, bool doPreconditioning){
  
  //TIMING VARIABLES  
  double t_begin = mytimer();  //start timing right away
  double t_dotProd = 0.0, t_WAXPBY = 0.0, t_SPMV = 0.0, t_MG = 0.0 , dummy_time = 0.0;
  double t_SYMGS = 0.0, t_restrict = 0.0, t_prolong = 0.0;

  //DATA VARIABLES
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

  //passed into kernels as const views of data
  const double * const * const * A_vals_const = A_vals;
  const local_int_t * const * const * A_inds_const = A_inds;
  const double * const * const * A_diags_const = A_diags;
  const local_int_t * A_nrows_const = A_nrows;
  const char * const * A_nnzs_const = A_nnzs;
  const unsigned char * const * A_colors_const = A_colors;

  unsigned int num_threads = omp_get_max_threads();
  std::cout << "THREAD POOL SIZE IS " << num_threads << ".\n";
  exec::static_thread_pool pool(num_threads);

#ifdef USE_GPU
  //scheduler for GPU execution
  nvexec::stream_context ctx;
  auto scheduler = ctx.get_scheduler();
#else
  //scheduler for CPU execution
  auto scheduler = pool.get_scheduler();
#endif
  
  if (!doPreconditioning && A.geom->rank == 0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
#endif

  sender auto pre_loop_work = schedule(scheduler)
  | WAXPBY(1, x_vals, 0, x_vals, p_vals)
  | SPMV(A_vals_const[0], p_vals, Ap_vals, A_inds_const[0], A_nnzs_const[0], A_nrows_const[0])
  | WAXPBY(1, b_vals, -1, Ap_vals, r_vals[0]) //WAXPBY: r = b - Ax (x stored in p)
  | COMPUTE_DOT_PRODUCT(r_vals[0], r_vals[0], normr_cpy)
  | then([=](){
    *normr_cpy = sqrt(*normr_cpy);
    *normr0_cpy = *normr_cpy; //record initial residual for convergence testing
  });
  sync_wait(std::move(pre_loop_work));

#ifdef HPCG_DEBUG
    if (A.geom->rank == 0) HPCG_fout << "Initial Residual = "<< *normr_cpy << std::endl;
#endif
  
  int k = 1;
  sender auto mg_point_0c = schedule(scheduler) | MGP0c();
  sender auto mg_point_1c = schedule(scheduler) | MGP1c();
  sender auto mg_point_2c = schedule(scheduler) | MGP2c();
  sender auto mg_point_4a = schedule(scheduler) | MGP4a();
  sender auto mg_point_5a = schedule(scheduler) | MGP5a();
  sender auto mg_point_6a = schedule(scheduler) | MGP6a();
  //ITERATION FOR FIRST LOOP
  MGP0a()
  MGP0b()
  sync_wait(mg_point_0c);
  MGP1a()
  MGP1b()
  sync_wait(mg_point_1c);
  MGP2a()
  MGP2b()
  sync_wait(mg_point_2c);
  MGP3a()
  MGP3b()
  sync_wait(mg_point_4a);
  MGP4b()
  sync_wait(mg_point_5a);
  MGP5b()
  sync_wait(mg_point_6a);
  MGP6b()

  sender auto rest_of_first_loop = schedule(scheduler)
    | WAXPBY(1, z_vals[0], 0, z_vals[0], p_vals)
    | COMPUTE_DOT_PRODUCT(r_vals[0], z_vals[0], rtz) //rtz = r'*z
    | SPMV(A_vals_const[0], p_vals, Ap_vals, A_inds_const[0], A_nnzs_const[0], A_nrows_const[0])
    | COMPUTE_DOT_PRODUCT(p_vals, Ap_vals, pAp) //alpha = p'*Ap
    | then([=](){ *alpha = *rtz/(*pAp); })
    | WAXPBY(1, x_vals, *alpha, p_vals, x_vals) //WAXPBY: x = x + alpha*p
    | WAXPBY(1, r_vals[0], -*alpha, Ap_vals, r_vals[0]) //WAXPBY: r = r - alpha*Ap
    | COMPUTE_DOT_PRODUCT(r_vals[0], r_vals[0], normr_cpy)
    | then([=](){ *normr_cpy = sqrt(*normr_cpy); });
    sync_wait(std::move(rest_of_first_loop));

#ifdef HPCG_DEBUG
    if(A.geom->rank == 0 && (1 % print_freq == 0 || 1 == max_iter))
      HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< *normr_cpy/(*normr0_cpy) << std::endl;
#endif
    niters = 1;

  sender auto rest_of_loop = schedule(scheduler)
    | then([=](){ *oldrtz = *rtz; })
    | COMPUTE_DOT_PRODUCT(r_vals[0], z_vals[0], rtz) //rtz = r'*z
    | then([=](){ *beta = *rtz/(*oldrtz); })
    | WAXPBY(1, z_vals[0], *beta, p_vals, p_vals) //WAXPBY: p = beta*p + z
    | SPMV(A_vals_const[0], p_vals, Ap_vals, A_inds_const[0], A_nnzs_const[0], A_nrows_const[0])
    | COMPUTE_DOT_PRODUCT(p_vals, Ap_vals, pAp) //alpha = p'*Ap
    | then([=](){ *alpha = *rtz/(*pAp); })
    | WAXPBY(1, x_vals, *alpha, p_vals, x_vals) //WAXPBY: x = x + alpha*p
    | WAXPBY(1, r_vals[0], -*alpha, Ap_vals, r_vals[0]) //WAXPBY: r = r - alpha*Ap
    | COMPUTE_DOT_PRODUCT(r_vals[0], r_vals[0], normr_cpy)
    | then([=](){ *normr_cpy = sqrt(*normr_cpy); });

  //start iterations
  //convergence check accepts an error of no more than 6 significant digits of tolerance
  for(int k = 2; k <= max_iter && *normr_cpy/(*normr0_cpy) > tolerance; k++){

    MGP0a()
    MGP0b()
    sync_wait(mg_point_0c);
    MGP1a()
    MGP1b()
    sync_wait(mg_point_1c);
    MGP2a()
    MGP2b()
    sync_wait(mg_point_2c);
    MGP3a()
    MGP3b()
    sync_wait(mg_point_4a);
    MGP4b()
    sync_wait(mg_point_5a);
    MGP5b()
    sync_wait(mg_point_6a);
    MGP6b()

    sync_wait(rest_of_loop);

#ifdef HPCG_DEBUG
    if(A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
      HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< *normr_cpy/(*normr0_cpy) << std::endl;
#endif
    niters = k;
  }

  normr = *normr_cpy;
  normr0 = *normr0_cpy;
  times[1] += t_dotProd; //dot-product time
  times[2] += t_WAXPBY; //WAXPBY time
  times[3] += t_SPMV; //SPMV time
  times[4] += 0.0; //AllReduce time
  times[5] += t_MG; //preconditioner apply time
  times[0] += mytimer() - t_begin;  //total time
  std::cout << "ADDITIONAL TIME DATA:\n";
  std::cout << "SYMGS Time : " << t_SYMGS << "\n";
  std::cout << "Restriction Time : " << t_restrict << "\n";
  std::cout << "Prolongation Time : " << t_prolong << "\n";
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