#include "metrics.h"
#include <matlib/matrix.h>
#include <stdio.h>

#define LARGE_N 1024
#define LARGE_N_WARM 30
#define LARGE_N_MEASURE 300

static void bench_transpose_fn(void *ctx) {
  Matrix *out = matlib_matrix_transpose((const Matrix *)ctx);
  metrics_do_not_optimise(out->data);
  matlib_matrix_free(out);
}

static size_t calc_iterations(size_t n, size_t large_n_iter) {
  if (LARGE_N <= n) {
    return large_n_iter;
  }
  size_t scale = LARGE_N / n;
  scale *= scale; // O(N^2) operation
  return scale * large_n_iter;
}

static double calc_bytes(size_t n) {
  return 2.0 * (double)n * (double)n * sizeof(float);
}

static void run_transpose(size_t n, const char *name) {
  Matrix *mat = matlib_matrix_rand(n, n, -1.0f, 1.0f);

  // No floating point arithmetics so no FLOPS
  MetricsConfig cfg = {
      .name = name,
      .warmup_iters = calc_iterations(n, LARGE_N_WARM),
      .measure_iters = calc_iterations(n, LARGE_N_MEASURE),
      .bytes = calc_bytes(n),
  };

  MetricsResult res = metrics_run(&cfg, bench_transpose_fn, mat);
  metrics_print_result(&res);

  matlib_matrix_free(mat);
}

int main(void) {
  printf("\nRunning: %s...\n", __FILE__);
  metrics_print_header();

  run_transpose(64, "transpose 64x64");
  run_transpose(512, "transpose 512x512");
  run_transpose(1024, "transpose 1024x1024");
  return 0;
}
