/*
 * ose.c - inclusive-range prime generator using a segmented 6-wheel sieve.
 *
 * Only values congruent to 1 or 5 modulo 6 are represented after handling
 * the primes 2 and 3 explicitly. The candidate set is stored as one bit per
 * value, while a conventional sieve supplies the base primes through
 * floor(sqrt(end)). This keeps memory proportional to the requested range
 * plus its square root rather than to every integer from zero through end.
 *
 * Standalone build:
 *   cc -O3 -std=c11 -Wall -Wextra -Wpedantic -o ose_cpu ose.c -lm
 *
 * Examples:
 *   ./ose_cpu 1 100
 *   ./ose_cpu 1 100 --cols 4
 *   ./ose_cpu 1 100 --json primes.json --no-list
 *   ./ose_cpu 1 100 --json -
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

typedef unsigned char byte_t;

/* Terminate on an unrecoverable validation or allocation failure. */
static void fail(const char *message) {
  (void)fprintf(stderr, "ERROR: %s\n", message);
  exit(EXIT_FAILURE);
}

/*
 * Keep raw allocation operations behind small, reviewed ownership helpers.
 * A zero-size allocation is rejected so callers never depend on the
 * implementation-defined result of malloc(0) or calloc(0, size).
 */
static void *checked_alloc(size_t size, const char *label) {
  if (size == 0U) {
    fail("internal zero-size allocation");
  }

  void *allocation = malloc(size); /* AES-SEC-001: reviewed allocation boundary. */
  if (allocation == NULL) {
    fail(label);
  }
  return allocation;
}

static void *checked_realloc(void *pointer, size_t size, const char *label) {
  if (size == 0U) {
    fail("internal zero-size reallocation");
  }

  void *allocation = realloc(pointer, size); /* AES-SEC-001: reviewed ownership boundary. */
  if (allocation == NULL) {
    fail(label);
  }
  return allocation;
}

static void checked_free(void *pointer) {
  free(pointer); /* AES-SEC-001: reviewed ownership release boundary. */
}

/* Parse a complete base-10 long; partial values such as "12x" are invalid. */
static long parse_long_arg(const char *text, const char *name) {
  char *end_pointer = NULL;
  errno = 0;
  long value = strtol(text, &end_pointer, 10);

  if (errno != 0 || end_pointer == text || *end_pointer != '\0') {
    (void)fprintf(stderr, "ERROR: invalid %s: %s\n", name, text);
    exit(2);
  }
  return value;
}

static int parse_positive_int_arg(const char *text, const char *name) {
  long value = parse_long_arg(text, name);
  if (value < 1L || value > INT_MAX) {
    (void)fprintf(stderr, "ERROR: %s must be in [1, %d]: %s\n", name,
                  INT_MAX, text);
    exit(2);
  }
  return (int)value;
}

/* C11's timespec_get avoids a platform-specific clock_gettime feature macro. */
static double now_milliseconds(void) {
  struct timespec timestamp;
  if (timespec_get(&timestamp, TIME_UTC) != TIME_UTC) {
    fail("could not read the system clock");
  }

  return (double)timestamp.tv_sec * 1000.0 +
         (double)timestamp.tv_nsec / 1000000.0;
}

/*
 * Return floor(sqrt(value)) and repair any floating-point rounding at the
 * integer boundary without multiplying values that could overflow long.
 */
static long integer_square_root(long value) {
  if (value <= 0L) {
    return 0L;
  }

  long root = (long)sqrtl((long double)value);
  while (root > value / root) {
    --root;
  }
  while (root < LONG_MAX) {
    long next = root + 1L;
    if (next > value / next) {
      break;
    }
    root = next;
  }
  return root;
}

/* Generate every base prime in [2, limit] using one byte per integer. */
static long *simple_sieve(long limit, size_t *prime_count) {
  *prime_count = 0U;
  if (limit < 2L) {
    return NULL;
  }
  if ((uintmax_t)limit >= (uintmax_t)SIZE_MAX) {
    fail("base-prime sieve is too large for this platform");
  }

  size_t value_count = (size_t)limit + 1U;
  byte_t *is_prime = checked_alloc(value_count, "out of memory for base sieve");
  for (size_t value = 0U; value < value_count; ++value) {
    is_prime[value] = 1U;
  }
  is_prime[0] = 0U;
  is_prime[1] = 0U;

  for (long prime = 2L; prime <= limit / prime; ++prime) {
    if (is_prime[(size_t)prime] == 0U) {
      continue;
    }

    for (long composite = prime * prime; composite <= limit;) {
      is_prime[(size_t)composite] = 0U;
      if (composite > limit - prime) {
        break;
      }
      composite += prime;
    }
  }

  size_t count = 0U;
  for (long value = 2L; value <= limit; ++value) {
    if (is_prime[(size_t)value] != 0U) {
      ++count;
    }
  }
  if (count > SIZE_MAX / sizeof(long)) {
    checked_free(is_prime);
    fail("base-prime array size overflow");
  }

  long *primes = checked_alloc(count * sizeof(*primes),
                               "out of memory for base primes");
  size_t output_index = 0U;
  for (long value = 2L; value <= limit; ++value) {
    if (is_prime[(size_t)value] != 0U) {
      primes[output_index++] = value;
    }
  }

  checked_free(is_prime);
  *prime_count = count;
  return primes;
}

/*
 * A wheel map describes the alternating +2/+4 candidate walk. first_step is
 * the distance from candidate zero to candidate one within each six-value
 * cycle: 4 for a first value congruent to 1, or 2 for one congruent to 5.
 */
typedef struct {
  long first;
  long end;
  int first_step;
  bool has_candidates;
} wheel6_map;

static int modulo_six(long value) {
  int remainder = (int)(value % 6L);
  return remainder < 0 ? remainder + 6 : remainder;
}

static wheel6_map build_wheel6(long start, long end) {
  wheel6_map wheel = {0L, end, 0, false};
  if (end < start) {
    return wheel;
  }

  long candidate = start;
  for (;;) {
    int remainder = modulo_six(candidate);
    if (remainder == 1 || remainder == 5) {
      wheel.first = candidate;
      wheel.first_step = remainder == 1 ? 4 : 2;
      wheel.has_candidates = true;
      return wheel;
    }
    if (candidate == end) {
      return wheel;
    }
    ++candidate;
  }
}

/* Count represented candidates without walking the complete input range. */
static size_t wheel6_candidate_count(const wheel6_map *wheel) {
  if (!wheel->has_candidates) {
    return 0U;
  }

  unsigned long distance = (unsigned long)(wheel->end - wheel->first);
  uintmax_t count = (uintmax_t)(distance / 6UL) * 2U + 1U;
  if (distance % 6UL >= (unsigned long)wheel->first_step) {
    ++count;
  }
  if (count > (uintmax_t)SIZE_MAX) {
    fail("candidate count exceeds addressable memory");
  }
  return (size_t)count;
}

/* Map a 6-wheel candidate to its bit index in constant time. */
static bool wheel6_index_of(const wheel6_map *wheel, long value,
                            size_t *index) {
  if (!wheel->has_candidates || value < wheel->first || value > wheel->end) {
    return false;
  }

  long distance = value - wheel->first;
  long cycle_offset = distance % 6L;
  if (cycle_offset != 0L && cycle_offset != (long)wheel->first_step) {
    return false;
  }

  uintmax_t candidate_index = (uintmax_t)(distance / 6L) * 2U;
  if (cycle_offset != 0L) {
    ++candidate_index;
  }
  if (candidate_index > (uintmax_t)SIZE_MAX) {
    fail("candidate index exceeds addressable memory");
  }

  *index = (size_t)candidate_index;
  return true;
}

#define BIT_GET(buffer, index)                                               \
  (((buffer)[(index) >> 3U] >> ((index)&7U)) & 1U)
#define BIT_CLEAR(buffer, index)                                             \
  ((buffer)[(index) >> 3U] &= (byte_t)~(1U << ((index)&7U)))

typedef struct {
  long *values;
  size_t count;
  size_t capacity;
} long_vector;

/* Append while checking both capacity doubling and byte-size arithmetic. */
static void vector_push(long_vector *vector, long value) {
  if (vector->count == vector->capacity) {
    size_t new_capacity = 256U;
    if (vector->capacity != 0U) {
      if (vector->capacity > SIZE_MAX / 2U) {
        fail("prime result capacity overflow");
      }
      new_capacity = vector->capacity * 2U;
    }
    if (new_capacity > SIZE_MAX / sizeof(*vector->values)) {
      fail("prime result byte-size overflow");
    }

    vector->values = checked_realloc(
        vector->values, new_capacity * sizeof(*vector->values),
        "out of memory for prime results");
    vector->capacity = new_capacity;
  }

  vector->values[vector->count++] = value;
}

/*
 * Mark composites in the requested range and append primes in ascending order.
 * start may be negative; the wheel begins at max(start, 5), so non-positive
 * inputs never enter the candidate set.
 */
static void sieve_collect(long start, long end, long_vector *output) {
  if (start <= 2L && end >= 2L) {
    vector_push(output, 2L);
  }
  if (start <= 3L && end >= 3L) {
    vector_push(output, 3L);
  }
  if (end < 5L) {
    return;
  }

  long candidate_start = start < 5L ? 5L : start;
  wheel6_map wheel = build_wheel6(candidate_start, end);
  size_t candidate_count = wheel6_candidate_count(&wheel);
  if (candidate_count == 0U) {
    return;
  }

  size_t byte_count = candidate_count / 8U;
  if (candidate_count % 8U != 0U) {
    ++byte_count;
  }
  byte_t *candidate_bits = checked_alloc(byte_count,
                                         "out of memory for candidate bits");
  for (size_t byte_index = 0U; byte_index < byte_count; ++byte_index) {
    candidate_bits[byte_index] = UCHAR_MAX;
  }

  size_t base_count = 0U;
  long limit = integer_square_root(end);
  long *base_primes = simple_sieve(limit, &base_count);

  for (size_t base_index = 0U; base_index < base_count; ++base_index) {
    long prime = base_primes[base_index];
    if (prime < 5L) {
      continue; /* Multiples of 2 and 3 are absent from the wheel. */
    }

    long first_multiple = prime * prime; /* prime <= floor(sqrt(end)). */
    if (candidate_start > first_multiple) {
      long remainder = candidate_start % prime;
      if (remainder == 0L) {
        first_multiple = candidate_start;
      } else {
        long adjustment = prime - remainder;
        if (candidate_start > LONG_MAX - adjustment) {
          continue; /* The next multiple lies outside the long domain. */
        }
        first_multiple = candidate_start + adjustment;
      }
    }

    for (long multiple = first_multiple; multiple <= end;) {
      size_t candidate_index = 0U;
      if (wheel6_index_of(&wheel, multiple, &candidate_index)) {
        BIT_CLEAR(candidate_bits, candidate_index);
      }

      if (multiple > end - prime) {
        break;
      }
      multiple += prime;
    }
  }

  long candidate = wheel.first;
  int step = wheel.first_step;
  for (size_t index = 0U; index < candidate_count; ++index) {
    if (BIT_GET(candidate_bits, index) != 0U) {
      vector_push(output, candidate);
    }
    if (index + 1U < candidate_count) {
      candidate += (long)step;
      step = step == 4 ? 2 : 4;
    }
  }

  checked_free(base_primes);
  checked_free(candidate_bits);
}

/* Return the interactive terminal width, or a stable noninteractive default. */
static int terminal_width(void) {
#if defined(__unix__) || defined(__APPLE__)
  struct winsize window;
  if (isatty(STDOUT_FILENO) != 0 &&
      ioctl(STDOUT_FILENO, TIOCGWINSZ, &window) == 0 && window.ws_col > 0U) {
    return (int)window.ws_col;
  }
#endif
  return 100;
}

/* Print column-major output so each visual column remains ascending. */
static void print_columns(const long *values, size_t count, int requested_columns) {
  if (count == 0U) {
    (void)putchar('\n');
    return;
  }

  long maximum = values[count - 1U];
  int value_width = 1;
  while (maximum >= 10L) {
    ++value_width;
    maximum /= 10L;
  }

  int column_width = value_width + 1;
  int columns = requested_columns;
  if (columns == 0) {
    int width = terminal_width();
    columns = width > column_width ? width / column_width : 1;
  }

  size_t column_count = (size_t)columns;
  size_t rows = count / column_count;
  if (count % column_count != 0U) {
    ++rows;
  }

  for (size_t row = 0U; row < rows; ++row) {
    for (size_t column = 0U; column < column_count; ++column) {
      size_t index = column * rows + row;
      if (index < count) {
        (void)printf("%*ld", value_width, values[index]);
      }
      if (column + 1U < column_count && index + rows < count) {
        (void)putchar(' ');
      }
    }
    (void)putchar('\n');
  }
}

/* Serialize a stable schema consumed by the repository's dice simulators. */
static int write_json(const char *path, long start, long end, const long *values,
                      size_t count) {
  bool use_stdout = path != NULL && strcmp(path, "-") == 0;
  FILE *stream = use_stdout ? stdout : fopen(path, "w");
  if (stream == NULL) {
    return -1;
  }

  (void)fprintf(stream, "{\n");
  (void)fprintf(stream, "  \"range\": {\"start\": %ld, \"end\": %ld},\n",
                start, end);
  (void)fprintf(stream, "  \"count\": %zu,\n", count);
  (void)fprintf(stream, "  \"primes\": [");
  for (size_t index = 0U; index < count; ++index) {
    if (index != 0U) {
      (void)fputc(',', stream);
    }
    if (index % 16U == 0U) {
      (void)fputc('\n', stream);
      (void)fputs("    ", stream);
    }
    (void)fprintf(stream, "%ld", values[index]);
  }
  if (count != 0U) {
    (void)fputc('\n', stream);
  }
  (void)fprintf(stream, "  ]\n}\n");

  if (use_stdout) {
    return fflush(stream) == 0 ? 0 : -1;
  }
  return fclose(stream) == 0 ? 0 : -1;
}

int main(int argument_count, char **argument_values) {
  if (argument_count < 3) {
    (void)fprintf(
        stderr,
        "Usage: %s START END [--cols N] [--json FILE|-] [--no-list]\n",
        argument_values[0]);
    return 2;
  }

  long start = parse_long_arg(argument_values[1], "START");
  long end = parse_long_arg(argument_values[2], "END");
  if (end < start) {
    fail("END is less than START");
  }

  int requested_columns = 0;
  const char *json_path = NULL;
  bool suppress_list = false;

  for (int index = 3; index < argument_count; ++index) {
    if (strcmp(argument_values[index], "--cols") == 0 &&
        index + 1 < argument_count) {
      requested_columns =
          parse_positive_int_arg(argument_values[++index], "--cols");
      continue;
    }
    if (strcmp(argument_values[index], "--json") == 0 &&
        index + 1 < argument_count) {
      json_path = argument_values[++index];
      continue;
    }
    if (strcmp(argument_values[index], "--no-list") == 0) {
      suppress_list = true;
      continue;
    }

    (void)fprintf(stderr, "Unknown or incomplete option: %s\n",
                  argument_values[index]);
    return 2;
  }

  /* JSON written to stdout must not be followed by human-readable columns. */
  if (json_path != NULL && strcmp(json_path, "-") == 0) {
    suppress_list = true;
  }

  double start_time = now_milliseconds();
  long_vector primes = {NULL, 0U, 0U};
  sieve_collect(start, end, &primes);
  double end_time = now_milliseconds();

  int exit_status = EXIT_SUCCESS;
  if (json_path != NULL &&
      write_json(json_path, start, end, primes.values, primes.count) != 0) {
    (void)fprintf(stderr, "ERROR: could not write JSON to %s\n", json_path);
    exit_status = EXIT_FAILURE;
  }
  if (!suppress_list) {
    print_columns(primes.values, primes.count, requested_columns);
  }

  (void)fprintf(stderr, "[ose] range=[%ld,%ld] count=%zu time=%.2f ms\n",
                start, end, primes.count, end_time - start_time);
  checked_free(primes.values);
  return exit_status;
}
