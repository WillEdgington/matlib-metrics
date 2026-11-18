CC = gcc
RELFLAGS = -O3 -march=native -ffast-math
CFLAGS = -Wall -Wextra -pedantic -std=c11 -D_POSIX_C_SOURCE=199309L -Iinclude -Ideps/matlib/include $(RELFLAGS) -MMD -MP
LDFLAGS = -Ldeps/matlib -lmatlib -lm

MATLIB_A = deps/matlib/libmatlib.a

SRC := $(wildcard src/*.c)
OBJ := $(SRC:.c=.o)

METRIC_SRC := $(wildcard metrics/*.c)
METRIC_BIN := $(METRIC_SRC:.c=)

DEPS := $(OBJ:.o=.d) $(METRIC_SRC:.c=.d)

.PHONY: all run setup_deps clean clean_deps

all: $(METRIC_BIN)

$(MATLIB_A):
		$(MAKE) -C deps/matlib

metrics/%: metrics/%.c $(OBJ) $(MATLIB_A)
		$(CC) $(CFLAGS) $< $(OBJ) $(LDFLAGS) -o $@

src/%.o: src/%.c
		$(CC) $(CFLAGS) -c -o $@ $<

run: all
		@for bin in $(METRIC_BIN); do ./$$bin; done

setup_deps:
		git submodule update --init

clean:
		rm -f $(OBJ) $(METRIC_BIN) $(DEPS)

clean_deps: clean
		$(MAKE) -C deps/matlib clean

-include $(DEPS)
