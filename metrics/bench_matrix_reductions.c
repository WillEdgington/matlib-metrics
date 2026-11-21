#include "metrics.h"
#include <matlib/matrix.h>
#include <stdbool.h>
#include <stdio.h>

#define LARGE_N 1024
#define LARGE_N_WARM 30
#define LARGE_N_MEASURE 300

static void bench_sum_fn(void *ctx) {
  float res = matlib_matrix_sum((Matrix *)ctx);
  metrics_do_not_optimise(&res);
}

static void bench_mean_fn(void *ctx) {
  float res = matlib_matrix_mean((Matrix *)ctx);
  metrics_do_not_optimise(&res);
}

static void bench_min_fn(void *ctx) {
  float res = matlib_matrix_min((Matrix *)ctx);
  metrics_do_not_optimise(&res);
}

static void bench_max_fn(void *ctx) {
  float res = matlib_matrix_max((Matrix *)ctx);
  metrics_do_not_optimise(&res);
}

static size_t calc_iterations(size_t n, size_t large_n_iter) {
  if (LARGE_N <= n) {
    return large_n_iter;
  }
  size_t scale = LARGE_N / n;
  scale *= scale; // all O(N^2) operation
  return scale * large_n_iter;
}

static double calc_flops(size_t n) {
  // sum & mean operations are (N^2) FLOPS (others are 0.0)
  return (double)n * (double)n;
}

static double calc_bytes(size_t n) {
  // all reduction operations involve 1 matrix (mat) of size N^2
  return (double)n * (double)n * sizeof(float);
}

static void run_benchmark(size_t n, const char *name, metrics_bench_fn fn,
                          bool has_flops) {
  Matrix *mat = matlib_matrix_rand(n, n, -1.0f, 1.0f);

  MetricsConfig cfg = {
      .name = name,
      .warmup_iters = calc_iterations(n, LARGE_N_WARM),
      .measure_iters = calc_iterations(n, LARGE_N_MEASURE),
      .bytes = calc_bytes(n),
  };

  if (has_flops) {
    cfg.flops = calc_flops(n);
  }

  MetricsResult res = metrics_run(&cfg, fn, mat);
  metrics_print_result(&res);

  matlib_matrix_free(mat);
}

int main(void) {
  printf("\nRunning: %s...\n", __FILE__);
  metrics_print_header();

  // matlib_matrix_sum(): N^2 FLOPS
  run_benchmark(64, "sum 64x64", bench_sum_fn, true);
  run_benchmark(512, "sum 512x512", bench_sum_fn, true);
  run_benchmark(1024, "sum 1024x1024", bench_sum_fn, true);

  // matlib_matrix_mean(): N^2 FLOPS
  run_benchmark(64, "mean 64x64", bench_mean_fn, true);
  run_benchmark(512, "mean 512x512", bench_mean_fn, true);
  run_benchmark(1024, "mean 1024x1024", bench_mean_fn, true);

  // matlib_matrix_min(): 0.0 FLOPS
  run_benchmark(64, "min 64x64", bench_min_fn, false);
  run_benchmark(512, "min 512x512", bench_min_fn, false);
  run_benchmark(1024, "min 1024x1024", bench_min_fn, false);

  // matlib_matrix_max(): 0.0 FLOPS
  run_benchmark(64, "max 64x64", bench_max_fn, false);
  run_benchmark(512, "max 512x512", bench_max_fn, false);
  run_benchmark(1024, "max 1024x1024", bench_max_fn, false);

  return 0;
}
