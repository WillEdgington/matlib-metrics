#include "metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_doubles(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

MetricsResult metrics_run(const MetricsConfig *cfg, metrics_bench_fn fn,
                          void *ctx) {
  MetricsResult res = {0};
  res.name = cfg->name;
  if (cfg->measure_iters == 0 || fn == NULL) {
    return res;
  }

  // contiguous buffer of time taken per fn iteration
  double *samples = (double *)malloc(cfg->measure_iters * sizeof(double));
  if (samples == NULL) {
    return res;
  }

  for (size_t i = 0; i < cfg->warmup_iters; i++) {
    fn(ctx);
  }

  // perform measured iterations, store run time in samples, accumalate total
  // time
  double sum_sec = 0.0;
  for (size_t i = 0; i < cfg->measure_iters; i++) {
    double t0 = now_sec();
    fn(ctx);
    double t1 = now_sec();

    double dt = t1 - t0;
    samples[i] = dt;
    sum_sec += dt;
  }

  // need min, mean, median so sort samples
  qsort(samples, cfg->measure_iters, sizeof(double), compare_doubles);

  res.min_sec = samples[0];
  res.mean_sec = sum_sec / (double)cfg->measure_iters;

  size_t mid = cfg->measure_iters / 2;
  if (cfg->measure_iters % 2 == 0) {
    // mean of two middle values
    res.median_sec = 0.5 * (samples[mid - 1] + samples[mid]);
  } else {
    res.median_sec = samples[mid];
  }
  free(samples);

  if (res.min_sec > 0.0) {
    res.gflops = (cfg->flops / res.min_sec) * 1e-9;
    res.gb_per_sec = (cfg->bytes / res.min_sec) * 1e-9;
  }
  if (cfg->bytes > 0.0) {
    res.arithmetic_intensity = cfg->flops / cfg->bytes;
  }

  return res;
}

void metrics_print_header(void) {
  printf("%-28s | %10s | %10s | %10s | %10s | %10s | %10s\n", "Benchmark",
         "Min (ms)", "Med (ms)", "Mean (ms)", "GFLOP/s", "GB/s", "FLOP/B");
  printf("-----------------------------+------------+------------+------------+"
         "------------+------------+------------\n");
}

void metrics_print_result(const MetricsResult *res) {
  printf("%-28s | %10.4f | %10.4f | %10.4f | %10.2f | %10.2f | %10.2f\n",
         res->name, res->min_sec * 1e3, res->median_sec * 1e3,
         res->mean_sec * 1e3, res->gflops, res->gb_per_sec,
         res->arithmetic_intensity);
}
