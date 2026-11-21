#include "metrics.h"
#include <matlib/matrix.h>
#include <matlib/random.h>
#include <stdio.h>

#define LARGE_N 1024
#define LARGE_N_WARM 30
#define LARGE_N_MEASURE 300

typedef struct {
  Matrix *mat;
  float scalar;
} MatscalarCtx;

static void bench_add_scalar_fn(void *ctx) {
  MatscalarCtx *mat_ctx = (MatscalarCtx *)ctx;
  Matrix *out = matlib_matrix_add_scalar(mat_ctx->mat, mat_ctx->scalar);
  metrics_do_not_optimise(out->data);
  matlib_matrix_free(out);
}

static void bench_sub_scalar_fn(void *ctx) {
  MatscalarCtx *mat_ctx = (MatscalarCtx *)ctx;
  Matrix *out = matlib_matrix_sub_scalar(mat_ctx->mat, mat_ctx->scalar);
  metrics_do_not_optimise(out->data);
  matlib_matrix_free(out);
}

static void bench_scale_fn(void *ctx) {
  MatscalarCtx *mat_ctx = (MatscalarCtx *)ctx;
  Matrix *out = matlib_matrix_scale(mat_ctx->mat, mat_ctx->scalar);
  metrics_do_not_optimise(out->data);
  matlib_matrix_free(out);
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
  // all scalar operations are (N^2) FLOPS (one per element)
  return (double)n * (double)n;
}

static double calc_bytes(size_t n) {
  // all scalar operations involve 2 matrice (mat, out) of size N^2
  return 2.0 * (double)n * (double)n * sizeof(float);
}

static void run_benchmark(size_t n, const char *name, metrics_bench_fn fn) {
  MatscalarCtx ctx = {
      .mat = matlib_matrix_rand(n, n, -1.0f, 1.0f),
      .scalar = matlib_random_uniform_float(-1.0f, 1.0f),
  };

  MetricsConfig cfg = {
      .name = name,
      .warmup_iters = calc_iterations(n, LARGE_N_WARM),
      .measure_iters = calc_iterations(n, LARGE_N_MEASURE),
      .flops = calc_flops(n),
      .bytes = calc_bytes(n),
  };

  MetricsResult res = metrics_run(&cfg, fn, &ctx);
  metrics_print_result(&res);

  matlib_matrix_free(ctx.mat);
}

int main(void) {
  printf("\nRunning: %s...\n", __FILE__);
  metrics_print_header();

  // matlib_matrix_add_scalar()
  run_benchmark(64, "add (scalar) 64x64", bench_add_scalar_fn);
  run_benchmark(512, "add (scalar) 512x512", bench_add_scalar_fn);
  run_benchmark(1024, "add (scalar) 1024x1024", bench_add_scalar_fn);

  // matlib_matrix_sub_scalar()
  run_benchmark(64, "sub (scalar) 64x64", bench_sub_scalar_fn);
  run_benchmark(512, "sub (scalar) 512x512", bench_sub_scalar_fn);
  run_benchmark(1024, "sub (scalar) 1024x1024", bench_sub_scalar_fn);

  // matlib_matrix_scale()
  run_benchmark(64, "scale 64x64", bench_scale_fn);
  run_benchmark(512, "scale 512x512", bench_scale_fn);
  run_benchmark(1024, "scale 1024x1024", bench_scale_fn);

  return 0;
}
