// cuda_sieve_mgpu.cu
// Multi-GPU segmented sieve (odd-only) with JSON or column output.
// Optimized for 4x Tesla K80 (SM 3.7).
//
// Build:
//   nvcc -O3 -arch=sm_37 -Xcompiler -fopenmp -o cuda_sieve_mgpu cuda_sieve_mgpu.cu
//
// Usage:
//   ./cuda_sieve_mgpu N                       # columns to terminal width
//   ./cuda_sieve_mgpu N --cols 6             # force 6 columns
//   ./cuda_sieve_mgpu N --json primes.json   # write JSON (no columns)
//   ./cuda_sieve_mgpu N --json -             # JSON to stdout
//   ./cuda_sieve_mgpu N --gpus 4 --seg 128M  # select #GPUs and segment size
//
// Notes:
//   - Stores only odds on device: candidate i -> value v = 2*i+1 (i>=1 corresponds to 3)
//   - Base primes up to floor(sqrt(N)) are computed on CPU (fast, tiny).
//   - Each GPU processes a disjoint slice of [3..N], segmented to bound memory.
//   - Results are merged on host in order, then printed or serialized.
//

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#ifndef TPB
#define TPB 256
#endif

#define CUDA_OK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

using u8 = unsigned char;

// ---------------- CPU base primes ----------------
static std::vector<int> simple_sieve(int limit) {
  if (limit < 1) return {};
  std::vector<u8> is(limit + 1, 1);
  is[0] = 0; if (limit >= 1) is[1] = 0;
  int r = (int)std::floor(std::sqrt((double)limit));
  for (int p = 2; p <= r; ++p) if (is[p]) {
    for (int q = p * p; q <= limit; q += p) is[q] = 0;
  }
  std::vector<int> primes;
  primes.reserve(limit / std::max(1.0, std::log((double)limit)));
  for (int i = 2; i <= limit; ++i) if (is[i]) primes.push_back(i);
  return primes;
}

// ---------------- Device kernel ------------------
// flags[k] == 1 means (2*k+1) is a candidate; 0 = composite.
// segment covers odd numbers [segL..segR] inclusive, segL & segR odd.
// Index mapping: val = 2*k+1  => k = (val-1)/2
__global__
void mark_multiples_odd(u8* flags, long long segL, long long segR, int p) {
  // First multiple >= segL:
  long long p2 = 1LL * p * p;
  long long start = (segL > p2 ? segL : p2);
  if ((start % 2) == 0) ++start;              // keep odd
  // advance start so that start ≡ 0 (mod p)
  long long rem = start % p;
  if (rem) start += (p - rem);
  if ((start % 2) == 0) start += p;           // ensure odd multiple of p

  // Grid-stride over multiples of p
  long long stride = 1LL * p * gridDim.x * blockDim.x * 2; // step by 2p per "full" cycle on odds
  long long first = start + 2LL * p * (blockIdx.x * blockDim.x + threadIdx.x);

  for (long long m = first; m <= segR; m += stride) {
    // m is odd multiple of p
    long long k = (m - segL) >> 1;            // index in flags for segment
    flags[k] = 0;
  }
}

// --------------- Utility: term width --------------
static int term_width() {
#if defined(__unix__) || defined(__APPLE__)
  #include <sys/ioctl.h>
  #include <unistd.h>
  struct winsize w;
  if (isatty(STDOUT_FILENO) && ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0)
    return (int)w.ws_col;
#endif
  return 100;
}

// --------------- Printing / JSON ------------------
static void print_columns(const std::vector<long long>& v, int cols) {
  if (v.empty()) { std::cout << "\n"; return; }
  // compute width
  long long mx = v.back();
  int w = 1; for (auto x = mx; x; x/=10) ++w;
  int colw = w + 1;
  if (cols <= 0) {
    int tw = term_width();
    cols = (tw > colw ? tw / colw : 1);
  }
  size_t n = v.size();
  size_t rows = (n + cols - 1) / cols;
  for (size_t r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      size_t i = c * rows + r;
      if (i < n) std::printf("%*lld", w, v[i]);
      if (c < cols - 1 && (c + 1) * rows + r < n) std::printf(" ");
    }
    std::printf("\n");
  }
}

static int write_json(const char* path, long long N, const std::vector<long long>& primes) {
  FILE* f = (path && std::string(path) == "-") ? stdout : std::fopen(path, "w");
  if (!f) return -1;
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"range\": {\"start\": 2, \"end\": %lld},\n", N);
  std::fprintf(f, "  \"count\": %zu,\n", primes.size());
  std::fprintf(f, "  \"primes\": [");
  for (size_t i = 0; i < primes.size(); ++i) {
    if (i % 16 == 0) std::fprintf(f, "\n    ");
    std::fprintf(f, "%lld", primes[i]);
    if (i + 1 < primes.size()) std::fputc(',', f);
  }
  if (!primes.empty()) std::fputc('\n', f);
  std::fprintf(f, "  ]\n}\n");
  if (f != stdout) std::fclose(f);
  return 0;
}

// --------------- Per-GPU worker -------------------
struct GpuSliceResult {
  std::vector<long long> primes;
  int device = -1;
};

struct WorkerArgs {
  int device;
  long long globalN;
  long long sliceL; // inclusive
  long long sliceR; // inclusive
  size_t segment_bytes; // bytes of flags per segment on device
  const int* base_primes; // host pointer
  int base_count;
};

static void gpu_worker(WorkerArgs args, std::promise<GpuSliceResult> prom) {
  CUDA_OK(cudaSetDevice(args.device));

  // Copy base primes to device
  int *d_base = nullptr;
  CUDA_OK(cudaMalloc(&d_base, args.base_count * sizeof(int)));
  CUDA_OK(cudaMemcpy(d_base, args.base_primes, args.base_count * sizeof(int), cudaMemcpyHostToDevice));

  // Output buffer on host
  std::vector<long long> out;
  out.reserve((size_t)( (args.sliceR - args.sliceL + 1) / std::log(std::max(3.0, (double)args.sliceR)) / 2 ));

  // Segment loop over odd numbers in [sliceL..sliceR]
  // Ensure segment bounds are odd
  auto make_odd = [](long long x){ return (x % 2 == 0) ? (x + 1) : x; };
  long long L = make_odd(args.sliceL);
  long long R = make_odd(args.sliceR);

  // How many odd numbers fit in segment_bytes?
  size_t max_flags = args.segment_bytes; // bytes of flags
  if (max_flags < 1024) max_flags = 1024; // floor
  // Each odd -> 1 byte flag. Segment span = 2*max_flags on values.
  long long seg_span_vals = 2LL * (long long)max_flags;

  u8* d_flags = nullptr;
  CUDA_OK(cudaMalloc(&d_flags, max_flags * sizeof(u8)));

  // Pre-create a stream for overlap (optional)
  cudaStream_t stream;
  CUDA_OK(cudaStreamCreate(&stream));

  for (long long segL = L; segL <= R; segL += seg_span_vals) {
    long long segR = std::min(segL + seg_span_vals - 2, R); // inclusive, keep odd
    size_t odd_count = (size_t)((segR - segL) / 2 + 1);     // number of odds in segment

    // Set all flags = 1
    CUDA_OK(cudaMemsetAsync(d_flags, 1, odd_count * sizeof(u8), stream));

    // Launch one kernel per base prime p >= 3
    int blocks = (int)((odd_count + TPB - 1) / TPB);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    for (int i = 0; i < args.base_count; ++i) {
      int p = args.base_primes[i];
      if (p == 2) continue;                     // odds only
      long long p2 = 1LL * p * p;
      if (p2 > segR) break;                     // nothing to mark beyond p^2
      mark_multiples_odd<<<blocks, TPB, 0, stream>>>(d_flags, segL, segR, p);
    }
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaStreamSynchronize(stream));

    // Copy flags back and collect primes
    std::vector<u8> h_flags(odd_count);
    CUDA_OK(cudaMemcpyAsync(h_flags.data(), d_flags, odd_count * sizeof(u8), cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));

    // Push primes from this segment
    // Also consider 2 if inside slice (handled once at the end in main)
    for (size_t k = 0; k < odd_count; ++k) {
      if (h_flags[k]) {
        long long val = segL + 2LL * (long long)k;
        if (val >= 3) out.push_back(val);
      }
    }
  }

  CUDA_OK(cudaStreamDestroy(stream));
  CUDA_OK(cudaFree(d_flags));
  CUDA_OK(cudaFree(d_base));

  GpuSliceResult res;
  res.device = args.device;
  std::sort(out.begin(), out.end());
  res.primes.swap(out);
  prom.set_value(std::move(res));
}

// --------------- CLI parsing helpers --------------
static long long parse_size_arg(const std::string& s) {
  // supports K/M/G suffix
  char suf = 0;
  long long v = 0;
  if (std::sscanf(s.c_str(), "%lld%c", &v, &suf) >= 1) {
    if (suf=='K'||suf=='k') v *= 1024LL;
    else if (suf=='M'||suf=='m') v *= 1024LL*1024LL;
    else if (suf=='G'||suf=='g') v *= 1024LL*1024LL*1024LL;
    return v;
  }
  std::fprintf(stderr, "Bad size: %s\n", s.c_str());
  std::exit(2);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s N [--gpus 4] [--seg 128M] [--cols N] [--json FILE|-]\n", argv[0]);
    return 2;
  }
  long long N = std::atoll(argv[1]);
  if (N < 2) { std::cout << "\n"; return 0; }

  int want_gpus = 4;
  long long seg_bytes = 128LL * 1024 * 1024; // default: 128 MB flags per GPU segment
  int forced_cols = 0;
  const char* json_path = nullptr;

  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--gpus") && i+1 < argc) { want_gpus = std::atoi(argv[++i]); continue; }
    if (!std::strcmp(argv[i], "--seg")  && i+1 < argc) { seg_bytes = parse_size_arg(argv[++i]); continue; }
    if (!std::strcmp(argv[i], "--cols") && i+1 < argc) { forced_cols = std::atoi(argv[++i]); continue; }
    if (!std::strcmp(argv[i], "--json") && i+1 < argc) { json_path  = argv[++i]; continue; }
    std::fprintf(stderr, "Unknown option: %s\n", argv[i]); return 2;
  }

  // Clamp to available devices
  int dev_count = 0;
  CUDA_OK(cudaGetDeviceCount(&dev_count));
  if (dev_count <= 0) { std::fprintf(stderr, "No CUDA devices found.\n"); return 2; }
  if (want_gpus > dev_count) want_gpus = dev_count;

  auto t0 = std::chrono::high_resolution_clock::now();

  // Base primes up to floor(sqrt(N))
  int limit = (int)std::floor(std::sqrt((double)N));
  std::vector<int> base = simple_sieve(limit);

  // Partition [3..N] across GPUs as contiguous slices
  long long L = 3;
  long long R = N;
  long long span = R - L + 1;
  long long per = span / want_gpus;
  long long rem = span % want_gpus;

  std::vector<std::future<GpuSliceResult>> futures;
  futures.reserve(want_gpus);

  const int* base_ptr = base.data();
  int base_count = (int)base.size();

  for (int g = 0; g < want_gpus; ++g) {
    long long a = L + g * per + std::min<long long>(g, rem);
    long long b = a + per - 1;
    if (g < rem) b += 1;
    if (b > R) b = R;

    WorkerArgs args;
    args.device = g;
    args.globalN = N;
    args.sliceL = a;
    args.sliceR = b;
    args.segment_bytes = (size_t)seg_bytes; // bytes of odd flags
    args.base_primes = base_ptr;
    args.base_count = base_count;

    std::promise<GpuSliceResult> prom;
    futures.emplace_back(prom.get_future());
    std::thread(gpu_worker, args, std::move(prom)).detach();
  }

  // Merge results in order (device slices are increasing by construction)
  std::vector<long long> primes;
  primes.reserve( (size_t)( N / std::max(2.0, std::log((double)N)) ) );

  // Include 2 if within range
  primes.push_back(2);

  for (auto& fut : futures) {
    GpuSliceResult r = fut.get();
    // r.primes already sorted
    primes.insert(primes.end(), r.primes.begin(), r.primes.end());
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Output
  if (json_path) {
    if (write_json(json_path, N, primes) != 0) {
      std::fprintf(stderr, "WARN: could not write JSON to %s\n", json_path);
    }
  } else {
    print_columns(primes, forced_cols);
  }

  std::fprintf(stderr, "[cuda_sieve] N=%lld  primes=%zu  gpus=%d  seg=%lld bytes  time=%.2f ms\n",
               N, primes.size(), want_gpus, seg_bytes, ms);
  return 0;
}