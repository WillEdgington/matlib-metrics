#include "metrics.h"
#include <matlib/tensor.h>
#include <stdbool.h>
#include <stdio.h>

#define LARGE_N (1024 * 1024)
#define LARGE_N_WARM 30
#define LARGE_N_MEASURE 300

static void bench_sum_fn(void *ctx) {
  float res = matlib_tensor_sum((Tensor *)ctx);
  metrics_do_not_optimise(&res);
}

static void bench_mean_fn(void *ctx) {
  float res = matlib_tensor_mean((Tensor *)ctx);
  metrics_do_not_optimise(&res);
}

static void bench_min_fn(void *ctx) {
  float res = matlib_tensor_min((Tensor *)ctx);
  metrics_do_not_optimise(&res);
}

static void bench_max_fn(void *ctx) {
  float res = matlib_tensor_max((Tensor *)ctx);
  metrics_do_not_optimise(&res);
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
  return (double)total_size * sizeof(float);
}

static void run_benchmark(size_t ndim, const size_t *shape, const char *name,
                          metrics_bench_fn fn, bool has_flops) {
  Tensor *tensor = matlib_tensor_rand(ndim, shape, -1.0f, 1.0f);
  size_t total_size = tensor->total_size;

  MetricsConfig cfg = {
      .name = name,
      .warmup_iters = calc_iterations(total_size, LARGE_N_WARM),
      .measure_iters = calc_iterations(total_size, LARGE_N_MEASURE),
      .bytes = calc_bytes(total_size),
  };

  if (has_flops) {
    cfg.flops = calc_flops(total_size);
  }

  MetricsResult res = metrics_run(&cfg, fn, tensor);
  metrics_print_result(&res);

  matlib_tensor_free(tensor);
}

int main(void) {
  printf("\nRunning: %s...\n", __FILE__);
  metrics_print_header();

  size_t sml_shape[] = {4, 32, 32};      // small
  size_t med_shape[] = {16, 64, 256};    // medium
  size_t lrg_shape[] = {16, 32, 32, 64}; // large

  // matlib_tensor_sum(): N FLOPS (where N = tensor->total_size)
  run_benchmark(3, sml_shape, "sum 4x32x32", bench_sum_fn, true);
  run_benchmark(3, med_shape, "sum 16x64x256", bench_sum_fn, true);
  run_benchmark(4, lrg_shape, "sum 16x32x32x64", bench_sum_fn, true);

  // matlib_tensor_mean(): N FLOPS (where N = tensor->total_size)
  run_benchmark(3, sml_shape, "mean 4x32x32", bench_mean_fn, true);
  run_benchmark(3, med_shape, "mean 16x64x256", bench_mean_fn, true);
  run_benchmark(4, lrg_shape, "mean 16x32x32x64", bench_mean_fn, true);

  // matlib_tensor_min(): 0.0 FLOPS
  run_benchmark(3, sml_shape, "min 4x32x32", bench_min_fn, false);
  run_benchmark(3, med_shape, "min 16x64x256", bench_min_fn, false);
  run_benchmark(4, lrg_shape, "min 16x32x32x64", bench_min_fn, false);

  // matlib_tensor_max(): 0.0 FLOPS
  run_benchmark(3, sml_shape, "max 4x32x32", bench_max_fn, false);
  run_benchmark(3, med_shape, "max 16x64x256", bench_max_fn, false);
  run_benchmark(4, lrg_shape, "max 16x32x32x64", bench_max_fn, false);

  return 0;
}
