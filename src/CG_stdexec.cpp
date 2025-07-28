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

  //double t_begin = mytimer();  //start timing right away
  //normr = 0.0;
  //double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  //double t_dotProd = 0.0, t_WAXPBY = 0.0, t_SPMV = 0.0, t_MG = 0.0 , dummy_time = 0.0;
  //double t_SYMGS = 0.0, t_restrict = 0.0, t_prolong = 0.0;
  const local_int_t nrow = A.localNumberOfRows;
  Vector &r = data.r; //residual vector
  Vector &z = data.z; //preconditioned residual vector
  //Vector &p = data.p; //direction vector (in MPI mode ncol>=nrow)
  //Vector &Ap = data.Ap;
  double local_result;

  //variables needed for MG computation
  std::vector<const SparseMatrix*> matrix_ptrs(NUM_MG_LEVELS);
  std::vector<const Vector*> res_ptrs(NUM_MG_LEVELS);
  std::vector<Vector*> z_ptrs(NUM_MG_LEVELS);
  std::vector<double*> Axfv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> rfv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> rcv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<local_int_t*> f2c_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> xfv_ptrs(NUM_MG_LEVELS - 1);
  std::vector<double*> xcv_ptrs(NUM_MG_LEVELS - 1);
  matrix_ptrs[0] = &A;
  res_ptrs[0] = &r;
  z_ptrs[0] = &z;
  for(int cnt = 1; cnt < NUM_MG_LEVELS; cnt++){
    matrix_ptrs[cnt] = matrix_ptrs[cnt - 1]->Ac;
    res_ptrs[cnt] = matrix_ptrs[cnt - 1]->mgData->rc;
    z_ptrs[cnt] = matrix_ptrs[cnt - 1]->mgData->xc;
  }
  for(int cnt = 0; cnt < NUM_MG_LEVELS - 1; cnt++){
    Axfv_ptrs[cnt] = matrix_ptrs[cnt]->mgData->Axf->values;
    rfv_ptrs[cnt] = res_ptrs[cnt]->values;
    rcv_ptrs[cnt] = matrix_ptrs[cnt]->mgData->rc->values;
    f2c_ptrs[cnt] = matrix_ptrs[cnt]->mgData->f2cOperator;
    xfv_ptrs[cnt] = z_ptrs[cnt]->values;
    xcv_ptrs[cnt] = matrix_ptrs[cnt]->mgData->xc->values;
  }

  //caching dereferenced pointers
  const auto& A0 = *matrix_ptrs[0];
  const auto& A1 = *matrix_ptrs[1];
  const auto& A2 = *matrix_ptrs[2];
  const auto& A3 = *matrix_ptrs[3];

  auto& r0 = *res_ptrs[0];
  auto& r1 = *res_ptrs[1];
  auto& r2 = *res_ptrs[2];
  auto& r3 = *res_ptrs[3];

  auto& z0 = *z_ptrs[0];
  auto& z1 = *z_ptrs[1];
  auto& z2 = *z_ptrs[2];
  auto& z3 = *z_ptrs[3];

  //used in dot product and WAXPBY calculations
  double * const rVals = r.values;
  double * const zVals = z.values;
  double * const pVals = p.values;
  double * const xVals = x.values;
  double * const ApVals = Ap.values;
  const double * const bVals = b.values;

  //for SPMV kernel
  //std::vector<double**> A_vals(NUM_MG_LEVELS);
  //std::vector<char*>  A_nnzs(NUM_MG_LEVELS);
  //std::vector<local_int_t**> A_inds(NUM_MG_LEVELS);
  //std::vector<local_int_t> A_nrows(NUM_MG_LEVELS);
  std::vector<double*> x_vals(NUM_MG_LEVELS);
  std::vector<double*> y_vals(NUM_MG_LEVELS);
  
  //populate values of SPMV kernel pointers
  for(int cnt = 0; cnt < NUM_MG_LEVELS - 2; cnt++){
    //A_vals[cnt] = matrix_ptrs[cnt]->matrixValues;
    //A_nnzs[cnt] = matrix_ptrs[cnt]->nonzerosInRow;
    //A_inds[cnt] = matrix_ptrs[cnt]->mtxIndL;
    //A_nrows[cnt] = matrix_ptrs[cnt]->localNumberOfRows;
    x_vals[cnt] = z_ptrs[cnt]->values;
    y_vals[cnt] = matrix_ptrs[cnt]->mgData->Axf->values;
  }

  //used for parallel SYMGS
  int *color = new int;

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

  //scheduler for SYMGS execution
  exec::static_thread_pool pool_single_thread(SINGLE_THREAD);
  auto scheduler_single_thread = pool_single_thread.get_scheduler();

  if (!doPreconditioning && A.geom->rank == 0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
#endif
std::cout << "1\n";
  sender auto pre_loop_work = schedule(scheduler)
  | WAXPBY(1, xVals, 0, xVals, pVals)
  | SPMV(A_vals[0], pVals, ApVals, A_inds[0], A_nnzs[0], A_nrows[0])
  | WAXPBY(1, bVals, -1, ApVals, rVals) //WAXPBY: r = b - Ax (x stored in p)
  | COMPUTE_DOT_PRODUCT(rVals, rVals, normr)
  | then([&](){
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif
    normr0 = normr; //record initial residual for convergence testing
  });
  sync_wait(std::move(pre_loop_work));
  std::cout << "2\n";
  int k = 1;
  //ITERATION FOR FIRST LOOP
  sender auto mg_downwards = schedule(scheduler)
    | COMPUTE_MG_STAGE1();
  sync_wait(std::move(mg_downwards));
  std::cout << "3\n";
  sender auto mg_upwards = schedule(scheduler)
    | COMPUTE_MG_STAGE2();
  sync_wait(std::move(mg_upwards));
std::cout << "4\n";
  sender auto rest_of_loop = schedule(scheduler)
    | WAXPBY(1, zVals, 0, zVals, pVals)
    | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz) //rtz = r'*z
    | SPMV(A_vals[0], pVals, ApVals, A_inds[0], A_nnzs[0], A_nrows[0])
    | COMPUTE_DOT_PRODUCT(pVals, ApVals, pAp) //alpha = p'*Ap
    | then([&](){ 
      alpha = rtz/pAp;
    })
    | WAXPBY(1, xVals, alpha, pVals, xVals) //WAXPBY: x = x + alpha*p
    | WAXPBY(1, rVals, -alpha, ApVals, rVals) //WAXPBY: r = r - alpha*Ap
    | COMPUTE_DOT_PRODUCT(rVals, rVals, normr)
    | then([&](){ 
      normr = sqrt(normr);
#ifdef HPCG_DEBUG
      if (A.geom->rank == 0 && (1 % print_freq == 0 || 1 == max_iter))
        HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
      niters = 1;
    });
    sync_wait(std::move(rest_of_loop));
std::cout << "5\n";
  //start iterations
  //convergence check accepts an error of no more than 6 significant digits of tolerance
  for(int k = 2; k <= max_iter && normr/normr0 > tolerance; k++){

    sender auto mg_downwards = schedule(scheduler)
    | COMPUTE_MG_STAGE1();
    sync_wait(std::move(mg_downwards));
  
    sender auto mg_upwards = schedule(scheduler)
    | COMPUTE_MG_STAGE2();
    sync_wait(std::move(mg_upwards));

    sender auto rest_of_loop = schedule(scheduler)
    | then([&](){ oldrtz = rtz; })
    | COMPUTE_DOT_PRODUCT(rVals, zVals, rtz) //rtz = r'*z
    | then([&](){ beta = rtz/oldrtz; })
    | WAXPBY(1, zVals, beta, pVals, pVals) //WAXPBY: p = beta*p + z
    | SPMV(A_vals[0], pVals, ApVals, A_inds[0], A_nnzs[0], A_nrows[0])
    | COMPUTE_DOT_PRODUCT(pVals, ApVals, pAp) //alpha = p'*Ap
    | then([&](){ alpha = rtz/pAp; })
    | WAXPBY(1, xVals, alpha, pVals, xVals) //WAXPBY: x = x + alpha*p
    | WAXPBY(1, rVals, -alpha, ApVals, rVals) //WAXPBY: r = r - alpha*Ap
    | COMPUTE_DOT_PRODUCT(rVals, rVals, normr)
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