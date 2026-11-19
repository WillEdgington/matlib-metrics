#ifndef MATLIB_METRICS_H
#define MATLIB_METRICS_H

#include <stddef.h>

typedef void (*metrics_bench_fn)(void *ctx);

typedef struct {
  const char *name;
  size_t warmup_iters;
  size_t measure_iters;
  double flops;
  double bytes;
} MetricsConfig;

typedef struct {
  const char *name;
  double min_sec;
  double median_sec;
  double mean_sec;
  double gflops;
  double gb_per_sec;
  double arithmetic_intensity;
} MetricsResult;

static inline void metrics_do_not_optimise(const void *ptr) {
  // force ptr to materialise in register/memory (preven dead-code elimination)
  __asm__ volatile("" : : "r,m"(ptr) : "memory");
}

MetricsResult metrics_run(const MetricsConfig *cfg, metrics_bench_fn fn,
                          void *ctx);

void metrics_print_header(void);
void metrics_print_result(const MetricsResult *res);

#endif