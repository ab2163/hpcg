#include "CG_stdexec.hpp"

int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
  const int max_iter, const double tolerance, int &niters, double &normr,  double &normr0,
  double *times, bool doPreconditioning){
  
  //TIMING VARIABLES  
  double t_begin = mytimer();  //start timing right away
  double t_dotProd = 0.0, t_WAXPBY = 0.0, t_SPMV = 0.0, t_MG = 0.0 , dummy_time = 0.0;
  double t_SYMGS = 0.0, t_restrict = 0.0, t_prolong = 0.0;

  //DATA VARIABLES
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  normr = 0.0;
  Vector &p = data.p; //direction vector (in MPI mode ncol>=nrow)
  Vector &Ap = data.Ap;
  //for the sparse matrix at different MG depths:
  std::vector<double**> A_vals(NUM_MG_LEVELS);
  std::vector<local_int_t**> A_inds(NUM_MG_LEVELS);
  std::vector<double**> A_diags(NUM_MG_LEVELS);
  std::vector<local_int_t> A_nrows(NUM_MG_LEVELS);
  std::vector<char*>  A_nnzs(NUM_MG_LEVELS);
  std::vector<unsigned char*>  A_colors(NUM_MG_LEVELS);
  //various vectors at different MG depths:
  std::vector<double*> r_vals(NUM_MG_LEVELS);
  std::vector<double*> z_vals(NUM_MG_LEVELS);
  std::vector<double*> Axfv_vals(NUM_MG_LEVELS - 1);
  std::vector<local_int_t*> f2c_vals(NUM_MG_LEVELS - 1);
  std::vector<double*> xcv_vals(NUM_MG_LEVELS - 1);
  std::vector<double*> rcv_vals(NUM_MG_LEVELS - 1);
  //objects used to initialise other data:
  std::vector<const SparseMatrix*> A_objs(NUM_MG_LEVELS);
  std::vector<Vector*> r_objs(NUM_MG_LEVELS);
  std::vector<Vector*> z_objs(NUM_MG_LEVELS);
  //variables used by main CG algorithm:
  double * const p_vals = p.values;
  double * const x_vals = x.values;
  double * const Ap_vals = Ap.values;
  const double * const b_vals = b.values;
  //used for parallel SYMGS:
  int *color = new int;
  *color = 0;

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

  //scheduler for CPU execution
  unsigned int num_threads = std::thread::hardware_concurrency();
  if(num_threads == 0){
    std::cerr << "ERROR: CANNOT DETERMINE THREAD POOL SIZE.\n";
    std::exit(EXIT_FAILURE);
  }
  else{
    std::cout << "THREAD POOL SIZE IS " << num_threads << ".\n";
  }
  exec::static_thread_pool pool(num_threads);
  auto scheduler = pool.get_scheduler();

  if (!doPreconditioning && A.geom->rank == 0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
#endif

  sender auto pre_loop_work = schedule(scheduler)
  | WAXPBY(1, x_vals, 0, x_vals, p_vals)
  | SPMV(A_vals[0], p_vals, Ap_vals, A_inds[0], A_nnzs[0], A_nrows[0])
  | WAXPBY(1, b_vals, -1, Ap_vals, r_vals[0]) //WAXPBY: r = b - Ax (x stored in p)
  | COMPUTE_DOT_PRODUCT(r_vals[0], r_vals[0], normr)
  | then([&](){
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif
    normr0 = normr; //record initial residual for convergence testing
  });
  sync_wait(std::move(pre_loop_work));
  
  int k = 1;
  //ITERATION FOR FIRST LOOP
  sync_wait(schedule(scheduler) | MGP0a());
  sync_wait(schedule(scheduler) | MGP0b());
  sync_wait(schedule(scheduler) | MGP1a());
  sync_wait(schedule(scheduler) | MGP1b());
  sync_wait(schedule(scheduler) | MGP2a());
  sync_wait(schedule(scheduler) | MGP2b());
  sync_wait(schedule(scheduler) | MGP3a());
  sync_wait(schedule(scheduler) | MGP3b());
  sync_wait(schedule(scheduler) | MGP4a());
  sync_wait(schedule(scheduler) | MGP4b());
  sync_wait(schedule(scheduler) | MGP5a());
  sync_wait(schedule(scheduler) | MGP5b());
  sync_wait(schedule(scheduler) | MGP6a());
  sync_wait(schedule(scheduler) | MGP6b());

  sender auto rest_of_loop = schedule(scheduler)
    | WAXPBY(1, z_vals[0], 0, z_vals[0], p_vals)
    | COMPUTE_DOT_PRODUCT(r_vals[0], z_vals[0], rtz) //rtz = r'*z
    | SPMV(A_vals[0], p_vals, Ap_vals, A_inds[0], A_nnzs[0], A_nrows[0])
    | COMPUTE_DOT_PRODUCT(p_vals, Ap_vals, pAp) //alpha = p'*Ap
    | then([&](){ 
      alpha = rtz/pAp;
    })
    | WAXPBY(1, x_vals, alpha, p_vals, x_vals) //WAXPBY: x = x + alpha*p
    | WAXPBY(1, r_vals[0], -alpha, Ap_vals, r_vals[0]) //WAXPBY: r = r - alpha*Ap
    | COMPUTE_DOT_PRODUCT(r_vals[0], r_vals[0], normr)
    | then([&](){ 
      normr = sqrt(normr);
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (1 % print_freq == 0 || 1 == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = 1;
    });
    sync_wait(std::move(rest_of_loop));

  //start iterations
  //convergence check accepts an error of no more than 6 significant digits of tolerance
  for(int k = 2; k <= max_iter && normr/normr0 > tolerance; k++){

    sync_wait(schedule(scheduler) | MGP0a());
    sync_wait(schedule(scheduler) | MGP0b());
    sync_wait(schedule(scheduler) | MGP1a());
    sync_wait(schedule(scheduler) | MGP1b());
    sync_wait(schedule(scheduler) | MGP2a());
    sync_wait(schedule(scheduler) | MGP2b());
    sync_wait(schedule(scheduler) | MGP3a());
    sync_wait(schedule(scheduler) | MGP3b());
    sync_wait(schedule(scheduler) | MGP4a());
    sync_wait(schedule(scheduler) | MGP4b());
    sync_wait(schedule(scheduler) | MGP5a());
    sync_wait(schedule(scheduler) | MGP5b());
    sync_wait(schedule(scheduler) | MGP6a());
    sync_wait(schedule(scheduler) | MGP6b());

    sender auto rest_of_loop = schedule(scheduler)
    | then([&](){ oldrtz = rtz; })
    | COMPUTE_DOT_PRODUCT(r_vals[0], z_vals[0], rtz) //rtz = r'*z
    | then([&](){ beta = rtz/oldrtz; })
    | WAXPBY(1, z_vals[0], beta, p_vals, p_vals) //WAXPBY: p = beta*p + z
    | SPMV(A_vals[0], p_vals, Ap_vals, A_inds[0], A_nnzs[0], A_nrows[0])
    | COMPUTE_DOT_PRODUCT(p_vals, Ap_vals, pAp) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; })
    | WAXPBY(1, x_vals, alpha, p_vals, x_vals) //WAXPBY: x = x + alpha*p
    | WAXPBY(1, r_vals[0], -alpha, Ap_vals, r_vals[0]) //WAXPBY: r = r - alpha*Ap
    | COMPUTE_DOT_PRODUCT(r_vals[0], r_vals[0], normr)
    | then([&](){ 
      normr = sqrt(normr); 
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = k;
    });
    sync_wait(std::move(rest_of_loop));
  }

  sender auto store_times = schedule(scheduler) | then([&](){
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
  });
  sync_wait(std::move(store_times));
  delete color;
  return 0;
}