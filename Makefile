CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c11 -D_POSIX_C_SOURCE=199309L -Iinclude -Ideps/matlib/include -MMD -MP
LDFLAGS = -Ldeps/matlib -lmatlib -lm

MATLIB_A = deps/matlib/libmatlib.a

SRC := $(wildcard src/*.c)
OBJ := $(SRC:.c=.o)

METRIC_SRC := $(wildcard metrics/*.c)
METRIC_BIN := $(METRIC_SRC:.c=)

DEPS := $(OBJ:.o=.d) $(METRIC_SRC:.c=.d)

RESULTS_DIR = results
DATE := $(shell date +%Y-%m-%d)
MATLIB_REV := $(shell git -C deps/matlib describe --tags --always --dirty 2>/dev/null || echo "unknown")
REPORT_FILE = $(RESULTS_DIR)/$(DATE)_$(MATLIB_REV).md

.PHONY: all run record setup_deps clean clean_deps

all: $(METRIC_BIN)

MATLIB_SRC := $(wildcard deps/matlib/src/*.c deps/matlib/include/matlib/*.h)

$(MATLIB_A): $(MATLIB_SRC)
		$(MAKE) -C deps/matlib

metrics/%: metrics/%.c $(OBJ) $(MATLIB_A)
		$(CC) $(CFLAGS) $< $(OBJ) $(LDFLAGS) -o $@

src/%.o: src/%.c
		$(CC) $(CFLAGS) -c -o $@ $<

run: all
		@for bin in $(METRIC_BIN); do ./$$bin; done

record: all
	@mkdir -p $(RESULTS_DIR)
	@{ \
		echo "# \`matlib\` Benchmark Baseline ($(DATE))"; \
		echo ""; \
		echo "- **Timestamp (UTC):** $$(date -u '+%Y-%m-%d %H:%M:%S UTC')"; \
		echo "- **matlib Revision:** \`$(MATLIB_REV)\`"; \
		echo "- **Compiler:** \`$(CC) $(CFLAGS)\`"; \
		echo "- **CPU:** $$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"; \
		echo "- **RAM:** $$(awk '/MemTotal/ {printf "%.1f GiB", $$2/1024/1024}' /proc/meminfo)"; \
		echo ""; \
		for bin in $(METRIC_BIN); do \
				METRICS_MD=1 ./$$bin; \
		done; \
	} > $(REPORT_FILE)
	@echo "Recorded benchmark results to $(REPORT_FILE)"

setup_deps:
		git submodule update --init

clean:
		rm -f $(OBJ) $(METRIC_BIN) $(DEPS)

clean_deps: clean
		$(MAKE) -C deps/matlib clean

-include $(DEPS)
