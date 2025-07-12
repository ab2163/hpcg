# 1 "CG_stdexec_str.cpp"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 453 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "CG_stdexec_str.cpp" 2
using stdexec::sender;
using stdexec::then;
using stdexec::schedule;
using stdexec::sync_wait;
using stdexec::bulk;
using stdexec::just;
using stdexec::continues_on;
using stdexec::let_value;
using exec::repeat_effect_until;
# 24 "CG_stdexec_str.cpp"
int indPC = 0;
const SparseMatrix* A_spmv;
Vector* x_spmv;
Vector* y_spmv;
bool disable_spmv;
# 47 "CG_stdexec_str.cpp"
inline void spmv_kernel(local_int_t i){
  if(disable_spmv){ return; }
  if(i >= A_spmv->localNumberOfRows){ return; }
  double sum = 0.0;
  double *cur_vals = A_spmv->matrixValues[i];
  local_int_t *cur_inds = A_spmv->mtxIndL[i];
  int cur_nnz = A_spmv->nonzerosInRow[i];
  double *xv = x_spmv->values;
  for(int j = 0; j < cur_nnz; j++)
    sum += cur_vals[j]*xv[cur_inds[j]];
  y_spmv->values[i] = sum;
}
# 102 "CG_stdexec_str.cpp"
int CG_stdexec(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
  const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
  double * times, bool doPreconditioning){

  double t_begin = mytimer();
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t_dotProd = 0.0, t_WAXPBY = 0.0, t_SPMV = 0.0, t_MG = 0.0, t_tmp = 0.0;
  double t_restrict = 0.0, t_prolong = 0.0;
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r;
  Vector & z = data.z;
  Vector & p = data.p;
  Vector & Ap = data.Ap;
  double local_result;


  std::vector<const SparseMatrix*> Aptrs(7);
  std::vector<const Vector*> rptrs(7);
  std::vector<Vector*> zptrs(7);
  std::vector<double*> Axfv_ptrs(7);
  std::vector<double*> rfv_ptrs(7);
  std::vector<double*> rcv_ptrs(7);
  std::vector<local_int_t*> f2c_ptrs(7);
  std::vector<double*> xfv_ptrs(7);
  std::vector<double*> xcv_ptrs(7);
  std::vector<bool> zerovector_flags(7, false);
  std::vector<bool> restrict_flags(7, true);
  std::vector<bool> prolong_flags(7, true);


  Aptrs[0] = &A;
  rptrs[0] = &r;
  zptrs[0] = &z;
  for(int cnt = 1; cnt < 4; cnt++){
    Aptrs[cnt] = Aptrs[cnt - 1]->Ac;
    rptrs[cnt] = Aptrs[cnt - 1]->mgData->rc;
    zptrs[cnt] = Aptrs[cnt - 1]->mgData->xc;
  }
  for(int cnt = 4; cnt < 7; cnt++){
    Aptrs[cnt] = Aptrs[7 - (cnt+1)];
    rptrs[cnt] = rptrs[7 - (cnt+1)];
    zptrs[cnt] = zptrs[7 - (cnt+1)];
  }


  for(int cnt = 0; cnt < 4 - 1; cnt++){
    Axfv_ptrs[cnt] = Aptrs[cnt]->mgData->Axf->values;
    rfv_ptrs[cnt] = rptrs[cnt]->values;
    rcv_ptrs[cnt] = Aptrs[cnt]->mgData->rc->values;
    f2c_ptrs[cnt] = Aptrs[cnt]->mgData->f2cOperator;
    xfv_ptrs[cnt] = zptrs[cnt]->values;
    xcv_ptrs[cnt] = Aptrs[cnt]->mgData->xc->values;
  }
  Axfv_ptrs[4 - 1] = Axfv_ptrs[4 - 2];
  rfv_ptrs[4 - 1] = rfv_ptrs[4 - 2];
  rcv_ptrs[4 - 1] = rcv_ptrs[4 - 2];
  f2c_ptrs[4 - 1] = f2c_ptrs[4 - 2];
  xfv_ptrs[4 - 1] = xfv_ptrs[4 - 2];
  xcv_ptrs[4 - 1] = xcv_ptrs[4 - 2];
  for(int cnt = 4; cnt < 7; cnt++){
    Axfv_ptrs[cnt] = Axfv_ptrs[7 - (cnt+1)];
    rfv_ptrs[cnt] = rfv_ptrs[7 - (cnt+1)];
    rcv_ptrs[cnt] = rcv_ptrs[7 - (cnt+1)];
    f2c_ptrs[cnt] = f2c_ptrs[7 - (cnt+1)];
    xfv_ptrs[cnt] = xfv_ptrs[7 - (cnt+1)];
    xcv_ptrs[cnt] = xcv_ptrs[7 - (cnt+1)];
  }


  for(int cnt = 0; cnt < 4; cnt++){
    zerovector_flags[cnt] = true;
  }


  for(int cnt = 0; cnt < 4 - 1; cnt++){
    restrict_flags[cnt] = false;
  }


  for(int cnt = 4; cnt < 7; cnt++){
    prolong_flags[cnt] = false;
  }


  double *rVals = r.values;
  double *zVals = z.values;
  double *pVals = p.values;
  double *xVals = x.values;
  double *bVals = b.values;
  double *ApVals = Ap.values;



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



  exec::inline_scheduler sched_inl;


  exec::static_thread_pool pool_single_thread(1);
  auto scheduler_single_thread = pool_single_thread.get_scheduler();

  if (!doPreconditioning && A.geom->rank == 0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;





  sender auto pre_loop_work = schedule(scheduler)
  | bulk(stdexec::par_unseq, nrow, [&](local_int_t i){ (pVals)[i] = (1)*(xVals)[i] + (0)*(xVals)[i]; })
  | then([&](){ t_tmp = mytimer(); A_spmv = &(A); x_spmv = &(p); if(!false){ y_spmv = &(Ap); } disable_spmv = (false); }) | bulk(stdexec::seq, false ? 0 : (A).localNumberOfRows, spmv_kernel) | then([&](){ t_SPMV += mytimer() - t_tmp; })
  | bulk(stdexec::par_unseq, nrow, [&](local_int_t i){ (rVals)[i] = (1)*(bVals)[i] + (-1)*(ApVals)[i]; })
  | then([&](){ (normr) = std::transform_reduce(std::execution::par, (rVals), (rVals) + nrow, (rVals), 0.0); })
  | then([&](){
    normr = sqrt(normr);



    normr0 = normr;
  });
  sync_wait(std::move(pre_loop_work));

  int k = 1;

  stdexec::sender auto sndr_first = stdexec::schedule(sched_inl)
  | then([&](){ t_tmp = mytimer(); }) | bulk(stdexec::par_unseq, prolong_flags[indPC] ? 0 : (*Aptrs[indPC]).mgData->rc->localLength, [&](int i){ if(prolong_flags[indPC]){ return; } if(i >= (A).mgData->rc->localLength){ return; } xfv_ptrs[(indPC)][f2c_ptrs[(indPC)][i]] += xcv_ptrs[(indPC)][i]; }) | then([&](){ t_prolong += mytimer() - t_tmp; }) | then([&](){ if(zerovector_flags[indPC]) ZeroVector(*zptrs[indPC]); }) | continues_on(scheduler_single_thread) | then([&](){ ComputeSYMGS_ref(*Aptrs[indPC], *rptrs[indPC], *zptrs[indPC]); }) | continues_on(scheduler) | then([&](){ t_tmp = mytimer(); A_spmv = &(*Aptrs[indPC]); x_spmv = &(*zptrs[indPC]); if(!restrict_flags[indPC]){ y_spmv = &(*((*Aptrs[indPC]).mgData->Axf)); } disable_spmv = (restrict_flags[indPC]); }) | bulk(stdexec::seq, restrict_flags[indPC] ? 0 : (*Aptrs[indPC]).localNumberOfRows, spmv_kernel) | then([&](){ t_SPMV += mytimer() - t_tmp; }) | then([&](){ t_tmp = mytimer(); }) | bulk(stdexec::par_unseq, restrict_flags[indPC] ? 0 : (*Aptrs[indPC]).mgData->rc->localLength, [&](int i){ if(restrict_flags[indPC]){ return; } if(i >= (*Aptrs[indPC]).mgData->rc->localLength){ return; } rcv_ptrs[(indPC)][i] = rfv_ptrs[(indPC)][f2c_ptrs[(indPC)][i]] - Axfv_ptrs[(indPC)][f2c_ptrs[(indPC)][i]]; }) | then([&](){ t_restrict += mytimer() - t_tmp; }) | then([&](){ indPC++; }) | exec::repeat_n(7)
  | bulk(stdexec::par_unseq, nrow, [&](local_int_t i){ (pVals)[i] = (1)*(zVals)[i] + (0)*(zVals)[i]; })
  | then([&](){ (rtz) = std::transform_reduce(std::execution::par, (rVals), (rVals) + nrow, (zVals), 0.0); });

  stdexec::sender auto sndr_second = stdexec::schedule(sched_inl)
  | then([&](){ t_tmp = mytimer(); }) | bulk(stdexec::par_unseq, prolong_flags[indPC] ? 0 : (*Aptrs[indPC]).mgData->rc->localLength, [&](int i){ if(prolong_flags[indPC]){ return; } if(i >= (A).mgData->rc->localLength){ return; } xfv_ptrs[(indPC)][f2c_ptrs[(indPC)][i]] += xcv_ptrs[(indPC)][i]; }) | then([&](){ t_prolong += mytimer() - t_tmp; }) | then([&](){ if(zerovector_flags[indPC]) ZeroVector(*zptrs[indPC]); }) | continues_on(scheduler_single_thread) | then([&](){ ComputeSYMGS_ref(*Aptrs[indPC], *rptrs[indPC], *zptrs[indPC]); }) | continues_on(scheduler) | then([&](){ t_tmp = mytimer(); A_spmv = &(*Aptrs[indPC]); x_spmv = &(*zptrs[indPC]); if(!restrict_flags[indPC]){ y_spmv = &(*((*Aptrs[indPC]).mgData->Axf)); } disable_spmv = (restrict_flags[indPC]); }) | bulk(stdexec::seq, restrict_flags[indPC] ? 0 : (*Aptrs[indPC]).localNumberOfRows, spmv_kernel) | then([&](){ t_SPMV += mytimer() - t_tmp; }) | then([&](){ t_tmp = mytimer(); }) | bulk(stdexec::par_unseq, restrict_flags[indPC] ? 0 : (*Aptrs[indPC]).mgData->rc->localLength, [&](int i){ if(restrict_flags[indPC]){ return; } if(i >= (*Aptrs[indPC]).mgData->rc->localLength){ return; } rcv_ptrs[(indPC)][i] = rfv_ptrs[(indPC)][f2c_ptrs[(indPC)][i]] - Axfv_ptrs[(indPC)][f2c_ptrs[(indPC)][i]]; }) | then([&](){ t_restrict += mytimer() - t_tmp; }) | then([&](){ indPC++; }) | exec::repeat_n(7)
  | then([&](){ oldrtz = rtz; })
  | then([&](){ (rtz) = std::transform_reduce(std::execution::par, (rVals), (rVals) + nrow, (zVals), 0.0); })
  | then([&](){ beta = rtz/oldrtz; })
  | bulk(stdexec::par_unseq, nrow, [&](local_int_t i){ (pVals)[i] = (1)*(zVals)[i] + (beta)*(pVals)[i]; });

  using loop_one_t = decltype(sndr_first);
  using loop_two_t = decltype(sndr_second);
  exec::variant_sender<loop_one_t, loop_two_t> switch_sndr = sndr_first;

  sender auto loop_work = let_value(stdexec::schedule(sched_inl), [&](){
    if(k == 2){
      switch_sndr.emplace<1>(sndr_second);
      return switch_sndr;
    }else{
      return switch_sndr;
    }
  })
  | then([&](){ t_tmp = mytimer(); A_spmv = &(A); x_spmv = &(p); if(!false){ y_spmv = &(Ap); } disable_spmv = (false); }) | bulk(stdexec::seq, false ? 0 : (A).localNumberOfRows, spmv_kernel) | then([&](){ t_SPMV += mytimer() - t_tmp; })
  | then([&](){ (pAp) = std::transform_reduce(std::execution::par, (pVals), (pVals) + nrow, (ApVals), 0.0); })
  | then([&](){ alpha = rtz/pAp; })
  | bulk(stdexec::par_unseq, nrow, [&](local_int_t i){ (xVals)[i] = (1)*(xVals)[i] + (alpha)*(pVals)[i]; })
  | bulk(stdexec::par_unseq, nrow, [&](local_int_t i){ (rVals)[i] = (1)*(rVals)[i] + (-alpha)*(ApVals)[i]; })
  | then([&](){ (normr) = std::transform_reduce(std::execution::par, (rVals), (rVals) + nrow, (rVals), 0.0); })
  | then([&](){
    normr = sqrt(normr);




    niters = k;
    k++;
    indPC = 0;
    return !(k <= max_iter && normr/normr0 > tolerance);
  })
  | repeat_effect_until();

  sync_wait(std::move(loop_work));

  sender auto store_times = schedule(scheduler) | then([&](){
    times[1] += t_dotProd;
    times[2] += t_WAXPBY;
    times[3] += t_SPMV;
    times[4] += 0.0;
    times[5] += t_MG;
    times[0] += mytimer() - t_begin;

    std::cout << "t_restrict: " << t_restrict << "\n";
    std::cout << "t_prolong: " << t_prolong << "\n";
  });
  sync_wait(std::move(store_times));
  return 0;
}
