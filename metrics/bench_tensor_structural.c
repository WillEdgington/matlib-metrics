#include "metrics.h"
#include <matlib/tensor.h>
#include <stdio.h>

#define LARGE_N (1024 * 1024)
#define LARGE_N_WARM 30
#define LARGE_N_MEASURE 300

typedef struct {
  Tensor *tensor;
  size_t new_ndim;
  const size_t *new_shape;
} TensorreshapeCtx;

typedef struct {
  Tensor *tensor;
  const size_t *axes;
} TensorpermuteCtx;

static void bench_reshape_fn(void *ctx) {
  TensorreshapeCtx *tensor_ctx = (TensorreshapeCtx *)ctx;
  Tensor *out = matlib_tensor_reshape(tensor_ctx->tensor, tensor_ctx->new_ndim,
                                      tensor_ctx->new_shape);
  metrics_do_not_optimise(out->data);
  matlib_tensor_free(out);
}

static void bench_permute_fn(void *ctx) {
  TensorpermuteCtx *tensor_ctx = (TensorpermuteCtx *)ctx;
  Tensor *out = matlib_tensor_permute(tensor_ctx->tensor, tensor_ctx->axes);
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

static double calc_bytes(size_t total_size) {
  return 2.0 * (double)total_size * sizeof(float);
}

static void run_reshape_benchmark(size_t ndim, const size_t *shape,
                                  size_t new_ndim, const size_t *new_shape,
                                  const char *name) {
  TensorreshapeCtx ctx = {
      .tensor = matlib_tensor_rand(ndim, shape, -1.0f, 1.0f),
      .new_ndim = new_ndim,
      .new_shape = new_shape,
  };

  size_t total_size = ctx.tensor->total_size;

  MetricsConfig cfg = {
      .name = name,
      .warmup_iters = calc_iterations(total_size, LARGE_N_WARM),
      .measure_iters = calc_iterations(total_size, LARGE_N_MEASURE),
      .flops = (double)0.0,
      .bytes = calc_bytes(total_size),
  };

  MetricsResult res = metrics_run(&cfg, bench_reshape_fn, &ctx);
  metrics_print_result(&res);

  matlib_tensor_free(ctx.tensor);
}

static void run_permute_benchmark(size_t ndim, const size_t *shape,
                                  const size_t *axes, const char *name) {
  TensorpermuteCtx ctx = {
      .tensor = matlib_tensor_rand(ndim, shape, -1.0f, 1.0f), .axes = axes};

  size_t total_size = ctx.tensor->total_size;

  MetricsConfig cfg = {
      .name = name,
      .warmup_iters = calc_iterations(total_size, LARGE_N_WARM),
      .measure_iters = calc_iterations(total_size, LARGE_N_MEASURE),
      .flops = (double)0.0,
      .bytes = calc_bytes(total_size),
  };

  MetricsResult res = metrics_run(&cfg, bench_permute_fn, &ctx);
  metrics_print_result(&res);

  matlib_tensor_free(ctx.tensor);
}

int main(void) {
  printf("\nRunning: %s...\n", __FILE__);
  metrics_print_header();

  // small
  size_t sml_shape[] = {4, 32, 32};
  size_t sml_new_shape[] = {64, 64};
  size_t sml_axes[] = {2, 0, 1};

  // medium
  size_t med_shape[] = {16, 64, 256};
  size_t med_new_shape[] = {512, 512};
  size_t med_axes[] = {2, 0, 1};

  // large
  size_t lrg_shape[] = {128, 128, 64};
  size_t lrg_new_shape[] = {1024, 1024};
  size_t lrg_axes[] = {2, 0, 1};

  // matlib_tensor_reshape()
  run_reshape_benchmark(3, sml_shape, 2, sml_new_shape,
                        "reshape 4x32x32->64x64");
  run_reshape_benchmark(3, med_shape, 2, med_new_shape,
                        "reshape 16x64x256->512x512");
  run_reshape_benchmark(3, lrg_shape, 2, lrg_new_shape,
                        "reshape 128x128x64->1024x1024");

  // matlib_tensor_permute()
  run_permute_benchmark(3, sml_shape, sml_axes, "permute 4x32x32 (2,0,1)");
  run_permute_benchmark(3, med_shape, med_axes, "permute 16x64x256 (2,0,1)");
  run_permute_benchmark(3, lrg_shape, lrg_axes, "permute 128x128x64 (2,0,1)");

  return 0;
}
