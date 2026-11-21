#include "metrics.h"
#include <matlib/matrix.h>
#include <stdio.h>

#define LARGE_N 1024
#define LARGE_N_WARM 30
#define LARGE_N_MEASURE 300

typedef struct {
  Matrix *a;
  Matrix *b;
} MatelemwiseCtx;

static void bench_add_fn(void *ctx) {
  MatelemwiseCtx *mat_ctx = (MatelemwiseCtx *)ctx;
  Matrix *out = matlib_matrix_add(mat_ctx->a, mat_ctx->b);
  metrics_do_not_optimise(out->data);
  matlib_matrix_free(out);
}

static void bench_sub_fn(void *ctx) {
  MatelemwiseCtx *mat_ctx = (MatelemwiseCtx *)ctx;
  Matrix *out = matlib_matrix_sub(mat_ctx->a, mat_ctx->b);
  metrics_do_not_optimise(out->data);
  matlib_matrix_free(out);
}

static void bench_mul_elementwise_fn(void *ctx) {
  MatelemwiseCtx *mat_ctx = (MatelemwiseCtx *)ctx;
  Matrix *out = matlib_matrix_mul_elementwise(mat_ctx->a, mat_ctx->b);
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
  // all elementwise operations are (N^2) FLOPS (one per element)
  return (double)n * (double)n;
}

static double calc_bytes(size_t n) {
  // all elementwise operations involve 3 matrice (a, b, c) of size N^2
  return 3.0 * (double)n * (double)n * sizeof(float);
}

static void run_benchmark(size_t n, const char *name, metrics_bench_fn fn) {
  MatelemwiseCtx ctx = {
      .a = matlib_matrix_rand(n, n, -1.0f, 1.0f),
      .b = matlib_matrix_rand(n, n, -1.0f, 1.0f),
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

  matlib_matrix_free(ctx.a);
  matlib_matrix_free(ctx.b);
}

int main(void) {
  printf("\nRunning: %s...\n", __FILE__);
  metrics_print_header();

  // matlib_matrix_add()
  run_benchmark(64, "add 64x64", bench_add_fn);
  run_benchmark(512, "add 512x512", bench_add_fn);
  run_benchmark(1024, "add 1024x1024", bench_add_fn);

  // matlib_matrix_sub()
  run_benchmark(64, "sub 64x64", bench_sub_fn);
  run_benchmark(512, "sub 512x512", bench_sub_fn);
  run_benchmark(1024, "sub 1024x1024", bench_sub_fn);

  // matlib_matrix_mul_elementwise()
  run_benchmark(64, "mul (Hadamard) 64x64", bench_mul_elementwise_fn);
  run_benchmark(512, "mul (Hadamard) 512x512", bench_mul_elementwise_fn);
  run_benchmark(1024, "mul (Hadamard) 1024x1024", bench_mul_elementwise_fn);

  return 0;
}
