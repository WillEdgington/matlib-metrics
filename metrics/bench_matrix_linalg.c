#include "metrics.h"
#include <matlib/matrix.h>
#include <stdio.h>

#define LARGE_N 1024
#define LARGE_N_WARM 10
#define LARGE_N_MEASURE 100

typedef struct {
  Matrix *a;
  Matrix *b;
} MatmulCtx;

static void bench_matmul_fn(void *ctx) {
  MatmulCtx *c = (MatmulCtx *)ctx;
  Matrix *out = matlib_matrix_mul(c->a, c->b);
  metrics_do_not_optimise(out->data);
  matlib_matrix_free(out);
}

static size_t calc_iterations(size_t n, size_t large_n_iter) {
  if (LARGE_N <= n) {
    return large_n_iter;
  }
  size_t scale = LARGE_N / n;
  scale *= scale * scale; // O(N^3) operation
  return scale * large_n_iter;
}

static double calc_flops(size_t n) {
  return 2.0 * (double)n * (double)n * (double)n;
}

static double calc_bytes(size_t n) {
  return 3.0 * (double)n * (double)n * sizeof(float);
}

static void run_matmul(size_t n, const char *name) {
  MatmulCtx ctx = {
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

  MetricsResult res = metrics_run(&cfg, bench_matmul_fn, &ctx);
  metrics_print_result(&res);

  matlib_matrix_free(ctx.a);
  matlib_matrix_free(ctx.b);
}

int main(void) {
  printf("\nRunning: %s...\n", __FILE__);
  metrics_print_header();

  run_matmul(64, "matmul 64x64");
  run_matmul(512, "matmul 512x512");
  run_matmul(1024, "matmul 1024x1024");
  return 0;
}
