# 0 "CG_stdexec.cpp"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "CG_stdexec.cpp"
# 1 "CG_stdexec.hpp" 1
# 24 "CG_stdexec.hpp"
using exec::repeat_n;
using stdexec::bulk;
using stdexec::continues_on;
using stdexec::just;
using stdexec::schedule;
using stdexec::sender;
using stdexec::sync_wait;
using stdexec::then;
# 132 "CG_stdexec.hpp"
int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
               const int max_iter, const double tolerance, int &niters,
               double &normr, double &normr0, double *times,
               bool doPreconditioning);
# 2 "CG_stdexec.cpp" 2

int CG_stdexec(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
               const int max_iter, const double tolerance, int &niters,
               double &normr, double &normr0, double *times,
               bool doPreconditioning) {

  double t_begin = mytimer();
  double t_dotProd = 0.0, t_WAXPBY = 0.0, t_SPMV = 0.0, t_MG = 0.0,
         dummy_time = 0.0;
  double t_SYMGS = 0.0, t_restrict = 0.0, t_prolong = 0.0;

  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  normr = 0.0;
  Vector &p = data.p;
  Vector &Ap = data.Ap;

  std::vector<double **> A_vals(4);
  std::vector<local_int_t **> A_inds(4);
  std::vector<double **> A_diags(4);
  std::vector<local_int_t> A_nrows(4);
  std::vector<char *> A_nnzs(4);
  std::vector<unsigned char *> A_colors(4);

  std::vector<double *> r_vals(4);
  std::vector<double *> z_vals(4);
  std::vector<double *> Axfv_vals(4 - 1);
  std::vector<local_int_t *> f2c_vals(4 - 1);
  std::vector<double *> xcv_vals(4 - 1);
  std::vector<double *> rcv_vals(4 - 1);

  std::vector<const SparseMatrix *> A_objs(4);
  std::vector<Vector *> r_objs(4);
  std::vector<Vector *> z_objs(4);

  double *const p_vals = p.values;
  double *const x_vals = x.values;
  double *const Ap_vals = Ap.values;
  const double *const b_vals = b.values;

  int *color = new int;

  A_objs[0] = &A;
  r_objs[0] = &data.r;
  z_objs[0] = &data.z;
  for (int depth = 1; depth < 4; depth++) {
    A_objs[depth] = A_objs[depth - 1]->Ac;
    r_objs[depth] = A_objs[depth - 1]->mgData->rc;
    z_objs[depth] = A_objs[depth - 1]->mgData->xc;
  }

  for (int depth = 0; depth < 4; depth++) {
    A_vals[depth] = A_objs[depth]->matrixValues;
    A_inds[depth] = A_objs[depth]->mtxIndL;
    A_diags[depth] = A_objs[depth]->matrixDiagonal;
    A_nrows[depth] = A_objs[depth]->localNumberOfRows;
    A_nnzs[depth] = A_objs[depth]->nonzerosInRow;
    A_colors[depth] = A_objs[depth]->colors;
    r_vals[depth] = r_objs[depth]->values;
    z_vals[depth] = z_objs[depth]->values;

    if (depth < 4 - 1) {
      Axfv_vals[depth] = A_objs[depth]->mgData->Axf->values;
      rcv_vals[depth] = A_objs[depth]->mgData->rc->values;
      f2c_vals[depth] = A_objs[depth]->mgData->f2cOperator;
      xcv_vals[depth] = A_objs[depth]->mgData->xc->values;
    }
  }

  local_int_t &nrow = A_nrows[0];

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    std::cerr << "ERROR: CANNOT DETERMINE THREAD POOL SIZE.\n";
    std::exit(EXIT_FAILURE);
  } else {
    std::cout << "THREAD POOL SIZE IS " << num_threads << ".\n";
  }
  exec::static_thread_pool pool(num_threads);
  auto scheduler = pool.get_scheduler();

  if (!doPreconditioning && A.geom->rank == 0)
    HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

  sender auto pre_loop_work =
      schedule(scheduler) |
      bulk(stdexec::par_unseq, nrow,
           [&](local_int_t i) {
             (p_vals)[i] = (1) * (x_vals)[i] + (0) * (x_vals)[i];
           }) |
      bulk(stdexec::par_unseq, (A_nrows[0]),
           [&](local_int_t i) {
             double sum = 0.0;
             for (int j = 0; j < (A_nnzs[0])[i]; j++) {
               sum += (A_vals[0])[i][j] * (p_vals)[(A_inds[0])[i][j]];
             }
             (Ap_vals)[i] = sum;
           }) |
      bulk(stdexec::par_unseq, nrow,
           [&](local_int_t i) {
             (r_vals[0])[i] = (1) * (b_vals)[i] + (-1) * (Ap_vals)[i];
           }) |
      then([&]() {
        local_result = 0.0;
        local_result =
            std::transform_reduce(std::execution::par_unseq, (r_vals[0]),
                                  (r_vals[0]) + nrow, (r_vals[0]), 0.0);
        MPI_Allreduce(&local_result, &(normr), 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
      }) |
      then([&]() {
        normr = sqrt(normr);

        normr0 = normr;
      });
  sync_wait(std::move(pre_loop_work));

  int k = 1;

  sync_wait(schedule(scheduler) | then([&]() { ZeroVector(*z_objs[0]); }) |
            then([=]() { *color = 0; }) |
            bulk(stdexec::par_unseq, (A_nrows[0]),
                 [=](local_int_t i) {
                   if ((A_colors[0])[i] == *color) {
                     const double currentDiagonal = (A_diags[0])[i][0];
                     double sum = (r_vals[0])[i];
                     for (int j = 0; j < (A_nnzs[0])[i]; j++) {
                       local_int_t curCol = (A_inds[0])[i][j];
                       sum -= (A_vals[0])[i][j] * (z_vals[0])[curCol];
                     }
                     sum += (z_vals[0])[i] * (A_diags[0]);
                     (z_vals[0])[i] = sum / (A_diags[0]);
                   }
                 }) |
            then([=]() { *color++; }) | repeat_n(8) | repeat_n(2) |
            bulk(stdexec::par_unseq, (A_nrows[0]),
                 [&](local_int_t i) {
                   double sum = 0.0;
                   for (int j = 0; j < (A_nnzs[0])[i]; j++) {
                     sum += (A_vals[0])[i][j] * (z_vals[0])[(A_inds[0])[i][j]];
                   }
                   (Axfv_vals[0])[i] = sum;
                 }) |
            bulk(stdexec::par_unseq, (*A_objs[0]).mgData->rc->localLength,
                 [&](int i) {
                   rcv_vals[(0)][i] = r_vals[(0)][f2c_vals[(0)][i]] -
                                      Axfv_vals[(0)][f2c_vals[(0)][i]];
                 }));
  sync_wait(schedule(scheduler) | then([&]() { ZeroVector(*z_objs[1]); }) |
            then([=]() { *color = 0; }) |
            bulk(stdexec::par_unseq, (A_nrows[1]),
                 [=](local_int_t i) {
                   if ((A_colors[1])[i] == *color) {
                     const double currentDiagonal = (A_diags[1])[i][0];
                     double sum = (r_vals[1])[i];
                     for (int j = 0; j < (A_nnzs[1])[i]; j++) {
                       local_int_t curCol = (A_inds[1])[i][j];
                       sum -= (A_vals[1])[i][j] * (z_vals[1])[curCol];
                     }
                     sum += (z_vals[1])[i] * (A_diags[1]);
                     (z_vals[1])[i] = sum / (A_diags[1]);
                   }
                 }) |
            then([=]() { *color++; }) | repeat_n(8) | repeat_n(2) |
            bulk(stdexec::par_unseq, (A_nrows[1]),
                 [&](local_int_t i) {
                   double sum = 0.0;
                   for (int j = 0; j < (A_nnzs[1])[i]; j++) {
                     sum += (A_vals[1])[i][j] * (z_vals[1])[(A_inds[1])[i][j]];
                   }
                   (Axfv_vals[1])[i] = sum;
                 }) |
            bulk(stdexec::par_unseq, (*A_objs[1]).mgData->rc->localLength,
                 [&](int i) {
                   rcv_vals[(1)][i] = r_vals[(1)][f2c_vals[(1)][i]] -
                                      Axfv_vals[(1)][f2c_vals[(1)][i]];
                 }));
  sync_wait(schedule(scheduler) | then([&]() { ZeroVector(*z_objs[2]); }) |
            then([=]() { *color = 0; }) |
            bulk(stdexec::par_unseq, (A_nrows[2]),
                 [=](local_int_t i) {
                   if ((A_colors[2])[i] == *color) {
                     const double currentDiagonal = (A_diags[2])[i][0];
                     double sum = (r_vals[2])[i];
                     for (int j = 0; j < (A_nnzs[2])[i]; j++) {
                       local_int_t curCol = (A_inds[2])[i][j];
                       sum -= (A_vals[2])[i][j] * (z_vals[2])[curCol];
                     }
                     sum += (z_vals[2])[i] * (A_diags[2]);
                     (z_vals[2])[i] = sum / (A_diags[2]);
                   }
                 }) |
            then([=]() { *color++; }) | repeat_n(8) | repeat_n(2) |
            bulk(stdexec::par_unseq, (A_nrows[2]),
                 [&](local_int_t i) {
                   double sum = 0.0;
                   for (int j = 0; j < (A_nnzs[2])[i]; j++) {
                     sum += (A_vals[2])[i][j] * (z_vals[2])[(A_inds[2])[i][j]];
                   }
                   (Axfv_vals[2])[i] = sum;
                 }) |
            bulk(stdexec::par_unseq, (*A_objs[2]).mgData->rc->localLength,
                 [&](int i) {
                   rcv_vals[(2)][i] = r_vals[(2)][f2c_vals[(2)][i]] -
                                      Axfv_vals[(2)][f2c_vals[(2)][i]];
                 }));
  sync_wait(schedule(scheduler) | then([&]() { ZeroVector(*z_objs[3]); }) |
            then([=]() { *color = 0; }) |
            bulk(stdexec::par_unseq, (A_nrows[3]),
                 [=](local_int_t i) {
                   if ((A_colors[3])[i] == *color) {
                     const double currentDiagonal = (A_diags[3])[i][0];
                     double sum = (r_vals[3])[i];
                     for (int j = 0; j < (A_nnzs[3])[i]; j++) {
                       local_int_t curCol = (A_inds[3])[i][j];
                       sum -= (A_vals[3])[i][j] * (z_vals[3])[curCol];
                     }
                     sum += (z_vals[3])[i] * (A_diags[3]);
                     (z_vals[3])[i] = sum / (A_diags[3]);
                   }
                 }) |
            then([=]() { *color++; }) | repeat_n(8) | repeat_n(2));
  sync_wait(
      schedule(scheduler) |
      bulk(stdexec::par_unseq, (*A_objs[2]).mgData->rc->localLength,
           [&](int i) { z_vals[(2)][f2c_vals[(2)][i]] += xcv_vals[(2)][i]; }) |
      then([=]() { *color = 0; }) |
      bulk(stdexec::par_unseq, (A_nrows[2]),
           [=](local_int_t i) {
             if ((A_colors[2])[i] == *color) {
               const double currentDiagonal = (A_diags[2])[i][0];
               double sum = (r_vals[2])[i];
               for (int j = 0; j < (A_nnzs[2])[i]; j++) {
                 local_int_t curCol = (A_inds[2])[i][j];
                 sum -= (A_vals[2])[i][j] * (z_vals[2])[curCol];
               }
               sum += (z_vals[2])[i] * (A_diags[2]);
               (z_vals[2])[i] = sum / (A_diags[2]);
             }
           }) |
      then([=]() { *color++; }) | repeat_n(8) | repeat_n(2));
  sync_wait(
      schedule(scheduler) |
      bulk(stdexec::par_unseq, (*A_objs[1]).mgData->rc->localLength,
           [&](int i) { z_vals[(1)][f2c_vals[(1)][i]] += xcv_vals[(1)][i]; }) |
      then([=]() { *color = 0; }) |
      bulk(stdexec::par_unseq, (A_nrows[1]),
           [=](local_int_t i) {
             if ((A_colors[1])[i] == *color) {
               const double currentDiagonal = (A_diags[1])[i][0];
               double sum = (r_vals[1])[i];
               for (int j = 0; j < (A_nnzs[1])[i]; j++) {
                 local_int_t curCol = (A_inds[1])[i][j];
                 sum -= (A_vals[1])[i][j] * (z_vals[1])[curCol];
               }
               sum += (z_vals[1])[i] * (A_diags[1]);
               (z_vals[1])[i] = sum / (A_diags[1]);
             }
           }) |
      then([=]() { *color++; }) | repeat_n(8) | repeat_n(2));
  sync_wait(
      schedule(scheduler) |
      bulk(stdexec::par_unseq, (*A_objs[0]).mgData->rc->localLength,
           [&](int i) { z_vals[(0)][f2c_vals[(0)][i]] += xcv_vals[(0)][i]; }) |
      then([=]() { *color = 0; }) |
      bulk(stdexec::par_unseq, (A_nrows[0]),
           [=](local_int_t i) {
             if ((A_colors[0])[i] == *color) {
               const double currentDiagonal = (A_diags[0])[i][0];
               double sum = (r_vals[0])[i];
               for (int j = 0; j < (A_nnzs[0])[i]; j++) {
                 local_int_t curCol = (A_inds[0])[i][j];
                 sum -= (A_vals[0])[i][j] * (z_vals[0])[curCol];
               }
               sum += (z_vals[0])[i] * (A_diags[0]);
               (z_vals[0])[i] = sum / (A_diags[0]);
             }
           }) |
      then([=]() { *color++; }) | repeat_n(8) | repeat_n(2));

  sender auto rest_of_loop =
      schedule(scheduler) |
      bulk(stdexec::par_unseq, nrow,
           [&](local_int_t i) {
             (p_vals)[i] = (1) * (z_vals[0])[i] + (0) * (z_vals[0])[i];
           }) |
      then([&]() {
        local_result = 0.0;
        local_result =
            std::transform_reduce(std::execution::par_unseq, (r_vals[0]),
                                  (r_vals[0]) + nrow, (z_vals[0]), 0.0);
        MPI_Allreduce(&local_result, &(rtz), 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
      }) |
      bulk(stdexec::par_unseq, (A_nrows[0]),
           [&](local_int_t i) {
             double sum = 0.0;
             for (int j = 0; j < (A_nnzs[0])[i]; j++) {
               sum += (A_vals[0])[i][j] * (p_vals)[(A_inds[0])[i][j]];
             }
             (Ap_vals)[i] = sum;
           }) |
      then([&]() {
        local_result = 0.0;
        local_result =
            std::transform_reduce(std::execution::par_unseq, (p_vals),
                                  (p_vals) + nrow, (Ap_vals), 0.0);
        MPI_Allreduce(&local_result, &(pAp), 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
      }) |
      then([&]() { alpha = rtz / pAp; }) |
      bulk(stdexec::par_unseq, nrow,
           [&](local_int_t i) {
             (x_vals)[i] = (1) * (x_vals)[i] + (alpha) * (p_vals)[i];
           }) |
      bulk(stdexec::par_unseq, nrow,
           [&](local_int_t i) {
             (r_vals[0])[i] = (1) * (r_vals[0])[i] + (-alpha) * (Ap_vals)[i];
           }) |
      then([&]() {
        local_result = 0.0;
        local_result =
            std::transform_reduce(std::execution::par_unseq, (r_vals[0]),
                                  (r_vals[0]) + nrow, (r_vals[0]), 0.0);
        MPI_Allreduce(&local_result, &(normr), 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
      }) |
      then([&]() {
        normr = sqrt(normr);

        niters = 1;
      });
  sync_wait(std::move(rest_of_loop));

  for (int k = 2; k <= max_iter && normr / normr0 > tolerance; k++) {

    sync_wait(schedule(scheduler) | then([&]() { ZeroVector(*z_objs[0]); }) |
              then([=]() { *color = 0; }) |
              bulk(stdexec::par_unseq, (A_nrows[0]),
                   [=](local_int_t i) {
                     if ((A_colors[0])[i] == *color) {
                       const double currentDiagonal = (A_diags[0])[i][0];
                       double sum = (r_vals[0])[i];
                       for (int j = 0; j < (A_nnzs[0])[i]; j++) {
                         local_int_t curCol = (A_inds[0])[i][j];
                         sum -= (A_vals[0])[i][j] * (z_vals[0])[curCol];
                       }
                       sum += (z_vals[0])[i] * (A_diags[0]);
                       (z_vals[0])[i] = sum / (A_diags[0]);
                     }
                   }) |
              then([=]() { *color++; }) | repeat_n(8) | repeat_n(2) |
              bulk(stdexec::par_unseq, (A_nrows[0]),
                   [&](local_int_t i) {
                     double sum = 0.0;
                     for (int j = 0; j < (A_nnzs[0])[i]; j++) {
                       sum +=
                           (A_vals[0])[i][j] * (z_vals[0])[(A_inds[0])[i][j]];
                     }
                     (Axfv_vals[0])[i] = sum;
                   }) |
              bulk(stdexec::par_unseq, (*A_objs[0]).mgData->rc->localLength,
                   [&](int i) {
                     rcv_vals[(0)][i] = r_vals[(0)][f2c_vals[(0)][i]] -
                                        Axfv_vals[(0)][f2c_vals[(0)][i]];
                   }));
    sync_wait(schedule(scheduler) | then([&]() { ZeroVector(*z_objs[1]); }) |
              then([=]() { *color = 0; }) |
              bulk(stdexec::par_unseq, (A_nrows[1]),
                   [=](local_int_t i) {
                     if ((A_colors[1])[i] == *color) {
                       const double currentDiagonal = (A_diags[1])[i][0];
                       double sum = (r_vals[1])[i];
                       for (int j = 0; j < (A_nnzs[1])[i]; j++) {
                         local_int_t curCol = (A_inds[1])[i][j];
                         sum -= (A_vals[1])[i][j] * (z_vals[1])[curCol];
                       }
                       sum += (z_vals[1])[i] * (A_diags[1]);
                       (z_vals[1])[i] = sum / (A_diags[1]);
                     }
                   }) |
              then([=]() { *color++; }) | repeat_n(8) | repeat_n(2) |
              bulk(stdexec::par_unseq, (A_nrows[1]),
                   [&](local_int_t i) {
                     double sum = 0.0;
                     for (int j = 0; j < (A_nnzs[1])[i]; j++) {
                       sum +=
                           (A_vals[1])[i][j] * (z_vals[1])[(A_inds[1])[i][j]];
                     }
                     (Axfv_vals[1])[i] = sum;
                   }) |
              bulk(stdexec::par_unseq, (*A_objs[1]).mgData->rc->localLength,
                   [&](int i) {
                     rcv_vals[(1)][i] = r_vals[(1)][f2c_vals[(1)][i]] -
                                        Axfv_vals[(1)][f2c_vals[(1)][i]];
                   }));
    sync_wait(schedule(scheduler) | then([&]() { ZeroVector(*z_objs[2]); }) |
              then([=]() { *color = 0; }) |
              bulk(stdexec::par_unseq, (A_nrows[2]),
                   [=](local_int_t i) {
                     if ((A_colors[2])[i] == *color) {
                       const double currentDiagonal = (A_diags[2])[i][0];
                       double sum = (r_vals[2])[i];
                       for (int j = 0; j < (A_nnzs[2])[i]; j++) {
                         local_int_t curCol = (A_inds[2])[i][j];
                         sum -= (A_vals[2])[i][j] * (z_vals[2])[curCol];
                       }
                       sum += (z_vals[2])[i] * (A_diags[2]);
                       (z_vals[2])[i] = sum / (A_diags[2]);
                     }
                   }) |
              then([=]() { *color++; }) | repeat_n(8) | repeat_n(2) |
              bulk(stdexec::par_unseq, (A_nrows[2]),
                   [&](local_int_t i) {
                     double sum = 0.0;
                     for (int j = 0; j < (A_nnzs[2])[i]; j++) {
                       sum +=
                           (A_vals[2])[i][j] * (z_vals[2])[(A_inds[2])[i][j]];
                     }
                     (Axfv_vals[2])[i] = sum;
                   }) |
              bulk(stdexec::par_unseq, (*A_objs[2]).mgData->rc->localLength,
                   [&](int i) {
                     rcv_vals[(2)][i] = r_vals[(2)][f2c_vals[(2)][i]] -
                                        Axfv_vals[(2)][f2c_vals[(2)][i]];
                   }));
    sync_wait(schedule(scheduler) | then([&]() { ZeroVector(*z_objs[3]); }) |
              then([=]() { *color = 0; }) |
              bulk(stdexec::par_unseq, (A_nrows[3]),
                   [=](local_int_t i) {
                     if ((A_colors[3])[i] == *color) {
                       const double currentDiagonal = (A_diags[3])[i][0];
                       double sum = (r_vals[3])[i];
                       for (int j = 0; j < (A_nnzs[3])[i]; j++) {
                         local_int_t curCol = (A_inds[3])[i][j];
                         sum -= (A_vals[3])[i][j] * (z_vals[3])[curCol];
                       }
                       sum += (z_vals[3])[i] * (A_diags[3]);
                       (z_vals[3])[i] = sum / (A_diags[3]);
                     }
                   }) |
              then([=]() { *color++; }) | repeat_n(8) | repeat_n(2));
    sync_wait(schedule(scheduler) |
              bulk(stdexec::par_unseq, (*A_objs[2]).mgData->rc->localLength,
                   [&](int i) {
                     z_vals[(2)][f2c_vals[(2)][i]] += xcv_vals[(2)][i];
                   }) |
              then([=]() { *color = 0; }) |
              bulk(stdexec::par_unseq, (A_nrows[2]),
                   [=](local_int_t i) {
                     if ((A_colors[2])[i] == *color) {
                       const double currentDiagonal = (A_diags[2])[i][0];
                       double sum = (r_vals[2])[i];
                       for (int j = 0; j < (A_nnzs[2])[i]; j++) {
                         local_int_t curCol = (A_inds[2])[i][j];
                         sum -= (A_vals[2])[i][j] * (z_vals[2])[curCol];
                       }
                       sum += (z_vals[2])[i] * (A_diags[2]);
                       (z_vals[2])[i] = sum / (A_diags[2]);
                     }
                   }) |
              then([=]() { *color++; }) | repeat_n(8) | repeat_n(2));
    sync_wait(schedule(scheduler) |
              bulk(stdexec::par_unseq, (*A_objs[1]).mgData->rc->localLength,
                   [&](int i) {
                     z_vals[(1)][f2c_vals[(1)][i]] += xcv_vals[(1)][i];
                   }) |
              then([=]() { *color = 0; }) |
              bulk(stdexec::par_unseq, (A_nrows[1]),
                   [=](local_int_t i) {
                     if ((A_colors[1])[i] == *color) {
                       const double currentDiagonal = (A_diags[1])[i][0];
                       double sum = (r_vals[1])[i];
                       for (int j = 0; j < (A_nnzs[1])[i]; j++) {
                         local_int_t curCol = (A_inds[1])[i][j];
                         sum -= (A_vals[1])[i][j] * (z_vals[1])[curCol];
                       }
                       sum += (z_vals[1])[i] * (A_diags[1]);
                       (z_vals[1])[i] = sum / (A_diags[1]);
                     }
                   }) |
              then([=]() { *color++; }) | repeat_n(8) | repeat_n(2));
    sync_wait(schedule(scheduler) |
              bulk(stdexec::par_unseq, (*A_objs[0]).mgData->rc->localLength,
                   [&](int i) {
                     z_vals[(0)][f2c_vals[(0)][i]] += xcv_vals[(0)][i];
                   }) |
              then([=]() { *color = 0; }) |
              bulk(stdexec::par_unseq, (A_nrows[0]),
                   [=](local_int_t i) {
                     if ((A_colors[0])[i] == *color) {
                       const double currentDiagonal = (A_diags[0])[i][0];
                       double sum = (r_vals[0])[i];
                       for (int j = 0; j < (A_nnzs[0])[i]; j++) {
                         local_int_t curCol = (A_inds[0])[i][j];
                         sum -= (A_vals[0])[i][j] * (z_vals[0])[curCol];
                       }
                       sum += (z_vals[0])[i] * (A_diags[0]);
                       (z_vals[0])[i] = sum / (A_diags[0]);
                     }
                   }) |
              then([=]() { *color++; }) | repeat_n(8) | repeat_n(2));

    sender auto rest_of_loop =
        schedule(scheduler) | then([&]() { oldrtz = rtz; }) | then([&]() {
          local_result = 0.0;
          local_result =
              std::transform_reduce(std::execution::par_unseq, (r_vals[0]),
                                    (r_vals[0]) + nrow, (z_vals[0]), 0.0);
          MPI_Allreduce(&local_result, &(rtz), 1, MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD);
        }) |
        then([&]() { beta = rtz / oldrtz; }) |
        bulk(stdexec::par_unseq, nrow,
             [&](local_int_t i) {
               (p_vals)[i] = (1) * (z_vals[0])[i] + (beta) * (p_vals)[i];
             }) |
        bulk(stdexec::par_unseq, (A_nrows[0]),
             [&](local_int_t i) {
               double sum = 0.0;
               for (int j = 0; j < (A_nnzs[0])[i]; j++) {
                 sum += (A_vals[0])[i][j] * (p_vals)[(A_inds[0])[i][j]];
               }
               (Ap_vals)[i] = sum;
             }) |
        then([&]() {
          local_result = 0.0;
          local_result =
              std::transform_reduce(std::execution::par_unseq, (p_vals),
                                    (p_vals) + nrow, (Ap_vals), 0.0);
          MPI_Allreduce(&local_result, &(pAp), 1, MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD);
        }) |
        then([&]() { alpha = rtz / pAp; }) |
        bulk(stdexec::par_unseq, nrow,
             [&](local_int_t i) {
               (x_vals)[i] = (1) * (x_vals)[i] + (alpha) * (p_vals)[i];
             }) |
        bulk(stdexec::par_unseq, nrow,
             [&](local_int_t i) {
               (r_vals[0])[i] = (1) * (r_vals[0])[i] + (-alpha) * (Ap_vals)[i];
             }) |
        then([&]() {
          local_result = 0.0;
          local_result =
              std::transform_reduce(std::execution::par_unseq, (r_vals[0]),
                                    (r_vals[0]) + nrow, (r_vals[0]), 0.0);
          MPI_Allreduce(&local_result, &(normr), 1, MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD);
        }) |
        then([&]() {
          normr = sqrt(normr);

          niters = k;
        });
    sync_wait(std::move(rest_of_loop));
  }

  sender auto store_times =
      schedule(scheduler) | then([&]() {
        times[1] += t_dotProd;
        times[2] += t_WAXPBY;
        times[3] += t_SPMV;
        times[4] += 0.0;
        times[5] += t_MG;
        times[0] += mytimer() - t_begin;
        std::cout << "ADDITIONAL TIME DATA:\n";
        std::cout << "SYMGS Time : " << t_SYMGS << "\n";
        std::cout << "Restriction Time : " << t_restrict << "\n";
        std::cout << "Prolongation Time : " << t_prolong << "\n";
      });
  sync_wait(std::move(store_times));
  delete color;
  return 0;
}
