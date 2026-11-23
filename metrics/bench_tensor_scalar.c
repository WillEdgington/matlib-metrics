#include "metrics.h"
#include <matlib/random.h>
#include <matlib/tensor.h>
#include <stdio.h>

#define LARGE_N (1024 * 1024)
#define LARGE_N_WARM 30
#define LARGE_N_MEASURE 300

typedef struct {
  Tensor *tensor;
  float scalar;
} TensorscalarCtx;

static void bench_add_scalar_fn(void *ctx) {
  TensorscalarCtx *tensor_ctx = (TensorscalarCtx *)ctx;
  Tensor *out =
      matlib_tensor_add_scalar(tensor_ctx->tensor, tensor_ctx->scalar);
  metrics_do_not_optimise(out->data);
  matlib_tensor_free(out);
}

static void bench_sub_scalar_fn(void *ctx) {
  TensorscalarCtx *tensor_ctx = (TensorscalarCtx *)ctx;
  Tensor *out =
      matlib_tensor_sub_scalar(tensor_ctx->tensor, tensor_ctx->scalar);
  metrics_do_not_optimise(out->data);
  matlib_tensor_free(out);
}

static void bench_scale_fn(void *ctx) {
  TensorscalarCtx *tensor_ctx = (TensorscalarCtx *)ctx;
  Tensor *out = matlib_tensor_scale(tensor_ctx->tensor, tensor_ctx->scalar);
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
  return 2.0 * (double)total_size * sizeof(float);
}

static void run_benchmark(size_t ndim, const size_t *shape, const char *name,
                          metrics_bench_fn fn) {
  TensorscalarCtx ctx = {
      .tensor = matlib_tensor_rand(ndim, shape, -1.0f, 1.0f),
      .scalar = matlib_random_uniform_float(-1.0f, 1.0f),
  };

  size_t total_size = ctx.tensor->total_size;

  MetricsConfig cfg = {
      .name = name,
      .warmup_iters = calc_iterations(total_size, LARGE_N_WARM),
      .measure_iters = calc_iterations(total_size, LARGE_N_MEASURE),
      .flops = calc_flops(total_size),
      .bytes = calc_bytes(total_size),
  };

  MetricsResult res = metrics_run(&cfg, fn, &ctx);
  metrics_print_result(&res);

  matlib_tensor_free(ctx.tensor);
}

int main(void) {
  printf("\nRunning: %s...\n", __FILE__);
  metrics_print_header();

  size_t sml_shape[] = {4, 32, 32};      // small
  size_t med_shape[] = {16, 64, 256};    // medium
  size_t lrg_shape[] = {16, 32, 32, 64}; // large

  // matlib_tensor_add_scalar()
  run_benchmark(3, sml_shape, "add (scalar) 4x32x32", bench_add_scalar_fn);
  run_benchmark(3, med_shape, "add (scalar) 16x64x256", bench_add_scalar_fn);
  run_benchmark(4, lrg_shape, "add (scalar) 16x32x32x64", bench_add_scalar_fn);

  // matlib_tensor_sub_scalar()
  run_benchmark(3, sml_shape, "sub (scalar) 4x32x32", bench_sub_scalar_fn);
  run_benchmark(3, med_shape, "sub (scalar) 16x64x256", bench_sub_scalar_fn);
  run_benchmark(4, lrg_shape, "sub (scalar) 16x32x32x64", bench_sub_scalar_fn);

  // matlib_tensor_scale()
  run_benchmark(3, sml_shape, "scale 4x32x32", bench_scale_fn);
  run_benchmark(3, med_shape, "scale 16x64x256", bench_scale_fn);
  run_benchmark(4, lrg_shape, "scale 16x32x32x64", bench_scale_fn);

  return 0;
}
