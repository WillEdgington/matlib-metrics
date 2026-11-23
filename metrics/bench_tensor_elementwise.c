#include "metrics.h"
#include <matlib/tensor.h>
#include <stdio.h>

#define LARGE_N (1024 * 1024)
#define LARGE_N_WARM 30
#define LARGE_N_MEASURE 300

typedef struct {
  Tensor *a;
  Tensor *b;
} TensorelemwiseCtx;

static void bench_add_fn(void *ctx) {
  TensorelemwiseCtx *tensor_ctx = (TensorelemwiseCtx *)ctx;
  Tensor *out = matlib_tensor_add(tensor_ctx->a, tensor_ctx->b);
  metrics_do_not_optimise(out->data);
  matlib_tensor_free(out);
}

static void bench_sub_fn(void *ctx) {
  TensorelemwiseCtx *tensor_ctx = (TensorelemwiseCtx *)ctx;
  Tensor *out = matlib_tensor_sub(tensor_ctx->a, tensor_ctx->b);
  metrics_do_not_optimise(out->data);
  matlib_tensor_free(out);
}

static void bench_mul_elementwise_fn(void *ctx) {
  TensorelemwiseCtx *tensor_ctx = (TensorelemwiseCtx *)ctx;
  Tensor *out = matlib_tensor_mul_elementwise(tensor_ctx->a, tensor_ctx->b);
  metrics_do_not_optimise(out->data);
  matlib_tensor_free(out);
}

static size_t calc_iterations(size_t total_size, size_t large_n_iter) {
  if (LARGE_N <= total_size) {
    return large_n_iter;
  }
  // O(N) operations, where n = total_size
  size_t scale = LARGE_N / total_size;
  return scale * large_n_iter;
}

static double calc_flops(size_t total_size) { return (double)total_size; }

static double calc_bytes(size_t total_size) {
  return 3.0 * (double)total_size * sizeof(float);
}

static void run_benchmark(size_t ndim, const size_t *shape, const char *name,
                          metrics_bench_fn fn) {
  TensorelemwiseCtx ctx = {
      .a = matlib_tensor_rand(ndim, shape, -1.0f, 1.0f),
      .b = matlib_tensor_rand(ndim, shape, -1.0f, 1.0f),
  };

  size_t total_size = ctx.a->total_size;

  MetricsConfig cfg = {
      .name = name,
      .warmup_iters = calc_iterations(total_size, LARGE_N_WARM),
      .measure_iters = calc_iterations(total_size, LARGE_N_MEASURE),
      .flops = calc_flops(total_size),
      .bytes = calc_bytes(total_size),
  };

  MetricsResult res = metrics_run(&cfg, fn, &ctx);
  metrics_print_result(&res);

  matlib_tensor_free(ctx.a);
  matlib_tensor_free(ctx.b);
}

int main(void) {
  printf("\nRunning: %s...\n", __FILE__);
  metrics_print_header();

  size_t sml_shape[] = {4, 32, 32};      // small
  size_t med_shape[] = {16, 64, 256};    // medium
  size_t lrg_shape[] = {16, 32, 32, 64}; // large

  // matlib_tensor_add()
  run_benchmark(3, sml_shape, "add 4x32x32", bench_add_fn);
  run_benchmark(3, med_shape, "add 16x64x256", bench_add_fn);
  run_benchmark(4, lrg_shape, "add 16x32x32x64", bench_add_fn);

  // matlib_tensor_sub()
  run_benchmark(3, sml_shape, "sub 4x32x32", bench_sub_fn);
  run_benchmark(3, med_shape, "sub 16x64x256", bench_sub_fn);
  run_benchmark(4, lrg_shape, "sub 16x32x32x64", bench_sub_fn);

  // matlib_tensor_mul_elementwise()
  run_benchmark(3, sml_shape, "mul 4x32x32", bench_mul_elementwise_fn);
  run_benchmark(3, med_shape, "mul 16x64x256", bench_mul_elementwise_fn);
  run_benchmark(4, lrg_shape, "mul 16x32x32x64", bench_mul_elementwise_fn);

  return 0;
}
