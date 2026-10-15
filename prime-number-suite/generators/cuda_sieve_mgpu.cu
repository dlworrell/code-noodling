/*
 * cuda_sieve_mgpu.cu - multi-GPU, odd-only segmented prime generator.
 *
 * The host computes base primes through floor(sqrt(N)), divides the odd
 * candidates in [3, N] into disjoint contiguous slices, and assigns one slice
 * to each selected CUDA device. Each device processes its slice in bounded
 * segments so memory use is controlled by --seg. Results are merged in slice
 * order and can be printed as columns or serialized for the dice simulators.
 *
 * Standalone build for the repository's Tesla K80 target:
 *   nvcc -O3 -std=c++17 -arch=sm_37 -o cuda_sieve_mgpu \
 *     cuda_sieve_mgpu.cu
 *
 * Examples:
 *   ./cuda_sieve_mgpu 50000000
 *   ./cuda_sieve_mgpu 50000000 --cols 6
 *   ./cuda_sieve_mgpu 50000000 --gpus 4 --seg 128M
 *   ./cuda_sieve_mgpu 50000000 --json primes_50M.json
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#ifndef TPB
#define TPB 256
#endif

using byte_t = unsigned char;

/* Convert every CUDA failure into a diagnostic containing the call site. */
static void check_cuda(cudaError_t result, const char *expression,
                       const char *file, int line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error for " << expression << " at " << file << ':'
              << line << ": " << cudaGetErrorString(result) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(expression)                                               \
  check_cuda((expression), #expression, __FILE__, __LINE__)

[[noreturn]] static void invalid_argument(const std::string &message) {
  std::cerr << "ERROR: " << message << '\n';
  std::exit(2);
}

/* Parse a complete signed decimal value and reject trailing characters. */
static long long parse_long_long(const char *text, const char *name) {
  char *end_pointer = nullptr;
  errno = 0;
  long long value = std::strtoll(text, &end_pointer, 10);
  if (errno != 0 || end_pointer == text || *end_pointer != '\0') {
    invalid_argument(std::string("invalid ") + name + ": " + text);
  }
  return value;
}

static int parse_positive_int(const char *text, const char *name) {
  long long value = parse_long_long(text, name);
  if (value < 1LL || value > static_cast<long long>(INT_MAX)) {
    invalid_argument(std::string(name) + " must be in [1, " +
                     std::to_string(INT_MAX) + "]: " + text);
  }
  return static_cast<int>(value);
}

/*
 * Parse byte counts with an optional binary K/M/G suffix. Overflow is checked
 * before multiplication and the result is constrained to both size_t and the
 * signed arithmetic used to advance segment bounds.
 */
static std::size_t parse_segment_bytes(const std::string &text) {
  if (text.empty() || text.front() == '-') {
    invalid_argument("invalid --seg value: " + text);
  }

  char *end_pointer = nullptr;
  errno = 0;
  unsigned long long base = std::strtoull(text.c_str(), &end_pointer, 10);
  if (errno != 0 || end_pointer == text.c_str()) {
    invalid_argument("invalid --seg value: " + text);
  }

  unsigned long long multiplier = 1ULL;
  if (*end_pointer != '\0') {
    if (end_pointer[1] != '\0') {
      invalid_argument("invalid --seg suffix: " + text);
    }
    switch (*end_pointer) {
      case 'K':
      case 'k':
        multiplier = 1024ULL;
        break;
      case 'M':
      case 'm':
        multiplier = 1024ULL * 1024ULL;
        break;
      case 'G':
      case 'g':
        multiplier = 1024ULL * 1024ULL * 1024ULL;
        break;
      default:
        invalid_argument("invalid --seg suffix: " + text);
    }
  }

  constexpr unsigned long long max_size_t =
      static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max());
  constexpr unsigned long long max_segment =
      static_cast<unsigned long long>(LLONG_MAX / 2LL);
  unsigned long long maximum = std::min(max_size_t, max_segment);
  if (base == 0ULL || base > maximum / multiplier) {
    invalid_argument("--seg is zero or exceeds the supported size: " + text);
  }

  unsigned long long bytes = base * multiplier;
  if (bytes < 1024ULL) {
    invalid_argument("--seg must reserve at least 1024 bytes");
  }
  return static_cast<std::size_t>(bytes);
}

/* Compute floor(sqrt(value)) and correct floating-point boundary rounding. */
static long long integer_square_root(long long value) {
  if (value <= 0LL) {
    return 0LL;
  }

  long long root = static_cast<long long>(std::sqrt(
      static_cast<long double>(value)));
  while (root > value / root) {
    --root;
  }
  while (root < LLONG_MAX) {
    long long next = root + 1LL;
    if (next > value / next) {
      break;
    }
    root = next;
  }
  return root;
}

/* Host-side sieve for the relatively small set of GPU marking primes. */
static std::vector<int> simple_sieve(int limit) {
  if (limit < 2) {
    return {};
  }

  std::vector<byte_t> is_prime(static_cast<std::size_t>(limit) + 1U, 1U);
  is_prime[0] = 0U;
  is_prime[1] = 0U;

  for (int prime = 2; prime <= limit / prime; ++prime) {
    if (is_prime[static_cast<std::size_t>(prime)] == 0U) {
      continue;
    }

    for (long long composite = 1LL * prime * prime; composite <= limit;
         composite += prime) {
      is_prime[static_cast<std::size_t>(composite)] = 0U;
    }
  }

  std::vector<int> primes;
  for (int value = 2; value <= limit; ++value) {
    if (is_prime[static_cast<std::size_t>(value)] != 0U) {
      primes.push_back(value);
    }
  }
  return primes;
}

/*
 * Mark odd multiples of one base prime in a segment. The kernel indexes the
 * sequence start, start+2p, start+4p, ... rather than multiplying an
 * unconstrained thread index into signed arithmetic.
 */
__global__ void mark_odd_multiples(byte_t *flags, long long segment_lower,
                                   long long segment_upper, int prime) {
  long long prime_squared = 1LL * prime * prime;
  long long first = segment_lower > prime_squared ? segment_lower
                                                   : prime_squared;

  long long remainder = first % prime;
  if (remainder != 0LL) {
    long long adjustment = static_cast<long long>(prime) - remainder;
    if (first > LLONG_MAX - adjustment) {
      return;
    }
    first += adjustment;
  }
  if ((first & 1LL) == 0LL) {
    if (first > LLONG_MAX - prime) {
      return;
    }
    first += prime;
  }
  if (first > segment_upper) {
    return;
  }

  unsigned long long prime_step =
      2ULL * static_cast<unsigned long long>(prime);
  unsigned long long multiple_count =
      static_cast<unsigned long long>(segment_upper - first) / prime_step + 1ULL;
  unsigned long long thread_index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long thread_count =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;

  for (unsigned long long offset = thread_index; offset < multiple_count;) {
    long long multiple = first + static_cast<long long>(offset * prime_step);
    std::size_t flag_index =
        static_cast<std::size_t>((multiple - segment_lower) / 2LL);
    flags[flag_index] = 0U;

    if (offset > ULLONG_MAX - thread_count) {
      break;
    }
    offset += thread_count;
  }
}

static int terminal_width() {
#if defined(__unix__) || defined(__APPLE__)
  struct winsize window;
  if (isatty(STDOUT_FILENO) != 0 &&
      ioctl(STDOUT_FILENO, TIOCGWINSZ, &window) == 0 && window.ws_col > 0U) {
    return static_cast<int>(window.ws_col);
  }
#endif
  return 100;
}

/* Print column-major output so each visual column remains ascending. */
static void print_columns(const std::vector<long long> &values,
                          int requested_columns) {
  if (values.empty()) {
    std::cout << '\n';
    return;
  }

  int value_width = 1;
  for (long long maximum = values.back(); maximum >= 10LL; maximum /= 10LL) {
    ++value_width;
  }

  int column_width = value_width + 1;
  int columns = requested_columns;
  if (columns == 0) {
    int width = terminal_width();
    columns = width > column_width ? width / column_width : 1;
  }

  std::size_t column_count = static_cast<std::size_t>(columns);
  std::size_t rows = values.size() / column_count;
  if (values.size() % column_count != 0U) {
    ++rows;
  }

  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t column = 0U; column < column_count; ++column) {
      std::size_t index = column * rows + row;
      if (index < values.size()) {
        std::cout << std::setw(value_width) << values[index];
      }
      if (column + 1U < column_count && index + rows < values.size()) {
        std::cout << ' ';
      }
    }
    std::cout << '\n';
  }
}

/* Serialize the same stable schema emitted by the CPU generator. */
static bool write_json(const std::string &path, long long upper_bound,
                       const std::vector<long long> &primes) {
  std::ofstream file;
  std::ostream *output = &std::cout;
  if (path != "-") {
    file.open(path);
    if (!file.is_open()) {
      return false;
    }
    output = &file;
  }

  *output << "{\n"
          << "  \"range\": {\"start\": 2, \"end\": " << upper_bound
          << "},\n"
          << "  \"count\": " << primes.size() << ",\n"
          << "  \"primes\": [";
  for (std::size_t index = 0U; index < primes.size(); ++index) {
    if (index != 0U) {
      *output << ',';
    }
    if (index % 16U == 0U) {
      *output << "\n    ";
    }
    *output << primes[index];
  }
  if (!primes.empty()) {
    *output << '\n';
  }
  *output << "  ]\n}\n";
  output->flush();
  return output->good();
}

struct gpu_worker_arguments {
  int device;
  long long slice_lower; /* Inclusive and odd. */
  long long slice_upper; /* Inclusive and odd. */
  std::size_t segment_bytes;
  const std::vector<int> *base_primes;
};

/* Process one disjoint odd-number slice on one CUDA device. */
static std::vector<long long> gpu_worker(gpu_worker_arguments arguments) {
  CUDA_CHECK(cudaSetDevice(arguments.device));

  byte_t *device_flags = nullptr;
  CUDA_CHECK(cudaMalloc(&device_flags, arguments.segment_bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<long long> output;
  long long segment_lower = arguments.slice_lower;
  for (;;) {
    unsigned long long available_odds =
        static_cast<unsigned long long>(arguments.slice_upper - segment_lower) /
            2ULL +
        1ULL;
    std::size_t odd_count = arguments.segment_bytes;
    if (available_odds < static_cast<unsigned long long>(odd_count)) {
      odd_count = static_cast<std::size_t>(available_odds);
    }

    long long segment_upper =
        segment_lower + 2LL * static_cast<long long>(odd_count - 1U);
    CUDA_CHECK(cudaMemsetAsync(device_flags, 1, odd_count, stream));

    std::size_t block_count = odd_count / static_cast<std::size_t>(TPB);
    if (odd_count % static_cast<std::size_t>(TPB) != 0U) {
      ++block_count;
    }
    block_count = std::max<std::size_t>(1U, block_count);
    block_count = std::min<std::size_t>(65535U, block_count);

    for (int prime : *arguments.base_primes) {
      if (prime == 2) {
        continue; /* Even candidates are not represented. */
      }
      if (1LL * prime * prime > segment_upper) {
        break;
      }
      mark_odd_multiples<<<static_cast<unsigned int>(block_count), TPB, 0,
                           stream>>>(device_flags, segment_lower, segment_upper,
                                    prime);
    }
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<byte_t> host_flags(odd_count);
    CUDA_CHECK(cudaMemcpyAsync(host_flags.data(), device_flags, odd_count,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (std::size_t index = 0U; index < odd_count; ++index) {
      if (host_flags[index] != 0U) {
        output.push_back(segment_lower + 2LL * static_cast<long long>(index));
      }
    }

    if (segment_upper == arguments.slice_upper) {
      break;
    }
    segment_lower = segment_upper + 2LL;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(device_flags));
  return output;
}

int main(int argument_count, char **argument_values) {
  if (argument_count < 2) {
    std::cerr << "Usage: " << argument_values[0]
              << " N [--gpus N] [--seg 128M] [--cols N] [--json FILE|-]\n";
    return 2;
  }

  long long upper_bound = parse_long_long(argument_values[1], "N");
  int requested_gpus = 4;
  std::size_t segment_bytes = 128U * 1024U * 1024U;
  int requested_columns = 0;
  std::string json_path;

  for (int index = 2; index < argument_count; ++index) {
    if (std::strcmp(argument_values[index], "--gpus") == 0 &&
        index + 1 < argument_count) {
      requested_gpus = parse_positive_int(argument_values[++index], "--gpus");
      continue;
    }
    if (std::strcmp(argument_values[index], "--seg") == 0 &&
        index + 1 < argument_count) {
      segment_bytes = parse_segment_bytes(argument_values[++index]);
      continue;
    }
    if (std::strcmp(argument_values[index], "--cols") == 0 &&
        index + 1 < argument_count) {
      requested_columns =
          parse_positive_int(argument_values[++index], "--cols");
      continue;
    }
    if (std::strcmp(argument_values[index], "--json") == 0 &&
        index + 1 < argument_count) {
      json_path = argument_values[++index];
      continue;
    }

    std::cerr << "Unknown or incomplete option: " << argument_values[index]
              << '\n';
    return 2;
  }

  auto start_time = std::chrono::steady_clock::now();
  std::vector<long long> primes;
  if (upper_bound >= 2LL) {
    primes.push_back(2LL);
  }

  int worker_count = 0;
  if (upper_bound >= 3LL) {
    long long base_limit = integer_square_root(upper_bound);
    if (base_limit > static_cast<long long>(INT_MAX)) {
      invalid_argument("N is too large for the current 32-bit base-prime index");
    }
    std::vector<int> base_primes = simple_sieve(static_cast<int>(base_limit));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) {
      std::cerr << "ERROR: no CUDA devices were found\n";
      return 2;
    }

    long long odd_candidate_count = (upper_bound - 3LL) / 2LL + 1LL;
    worker_count = std::min(requested_gpus, device_count);
    if (static_cast<long long>(worker_count) > odd_candidate_count) {
      worker_count = static_cast<int>(odd_candidate_count);
    }

    long long candidates_per_worker = odd_candidate_count / worker_count;
    long long remainder = odd_candidate_count % worker_count;
    long long next_candidate_index = 0LL;
    std::vector<std::future<std::vector<long long>>> futures;
    futures.reserve(static_cast<std::size_t>(worker_count));

    for (int device = 0; device < worker_count; ++device) {
      long long slice_count =
          candidates_per_worker + (device < remainder ? 1LL : 0LL);
      long long slice_first_index = next_candidate_index;
      long long slice_last_index = slice_first_index + slice_count - 1LL;
      next_candidate_index += slice_count;

      gpu_worker_arguments arguments{
          device,
          3LL + 2LL * slice_first_index,
          3LL + 2LL * slice_last_index,
          segment_bytes,
          &base_primes,
      };
      futures.push_back(
          std::async(std::launch::async, gpu_worker, arguments));
    }

    /* Futures are stored in slice order, so concatenation preserves sorting. */
    for (auto &future : futures) {
      std::vector<long long> slice_primes = future.get();
      primes.insert(primes.end(), slice_primes.begin(), slice_primes.end());
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  double elapsed_milliseconds =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  bool output_ok = true;
  if (!json_path.empty()) {
    output_ok = write_json(json_path, upper_bound, primes);
    if (!output_ok) {
      std::cerr << "ERROR: could not write JSON to " << json_path << '\n';
    }
  } else {
    print_columns(primes, requested_columns);
  }

  std::cerr << "[cuda_sieve] N=" << upper_bound << " primes=" << primes.size()
            << " gpus=" << worker_count << " seg=" << segment_bytes
            << " bytes time=" << std::fixed << std::setprecision(2)
            << elapsed_milliseconds << " ms\n";
  return output_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
