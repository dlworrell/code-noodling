// dice_cpu.cpp — unbiased dice for any n, with optional prime-seeded determinism.
// Build: add_executable(dice_cpu dice_cpu.cpp)  (CMake snippet below)
// Usage examples:
//   ./dice_cpu --faces 6 --count 10000 --chi
//   ./dice_cpu --spec "3d6+2" --count 5000 --use-prime-seeds primes.json --chi
//   ./dice_cpu --faces 20 --count 20000 --use-prime-seeds primes.json --csv rolls.csv --chi
//   ./dice_cpu --spec "4d8-1" --count 1000 --log-json rolls.json

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct RollSpec { int N=1, M=6, K=0; }; // N dice (M faces) + K

// -------- tiny JSON loader for ["primes": [...]] without dependencies -----------
static std::vector<uint64_t> load_primes_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::cerr << "WARN: cannot open primes file: " << path << "\n"; return {}; }
    std::string s((std::istreambuf_iterator<char>(in)), {});
    std::vector<uint64_t> p; p.reserve(1000);
    auto pos = s.find("\"primes\"");
    if (pos == std::string::npos) { std::cerr << "WARN: no \"primes\" key in " << path << "\n"; return p; }
    pos = s.find('[', pos); if (pos == std::string::npos) return p;
    auto end = s.find(']', pos); if (end == std::string::npos) return p;
    std::string arr = s.substr(pos+1, end-pos-1);
    size_t i=0;
    while (i < arr.size()) {
        while (i < arr.size() && (arr[i]==' '||arr[i]==','||arr[i]=='\n'||arr[i]=='\r'||arr[i]=='\t')) ++i;
        size_t j=i;
        while (j < arr.size() && std::isdigit((unsigned char)arr[j])) ++j;
        if (j>i) p.push_back(std::stoull(arr.substr(i, j-i)));
        i = (j==i? i+1 : j);
    }
    return p;
}

// -------- RNG + bias-free sampler (Lemire) --------------------------------------
static thread_local std::mt19937_64 g_rng{0x9e3779b97f4a7c15ULL}; // default seed

static inline uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// seed RNG deterministically from a prime
static inline void seed_from_prime(uint64_t p) { g_rng.seed(splitmix64(p)); }

// unbiased integer in [0..n-1] for any n (Lemire + rejection)
static inline uint64_t uniform_u64_unbiased(uint64_t n){
    if (n == 0) return 0;
    while (true) {
        uint64_t x = g_rng();                       // 64 random bits
        __uint128_t m = (__uint128_t)x * (__uint128_t)n;
        uint64_t l = (uint64_t)m;                   // low 64
        if (l < n) {
            uint64_t t = (-n) % n;
            if (l < t) continue;                    // reject region
        }
        return (uint64_t)(m >> 64);                 // upper 64 -> uniform [0..n-1]
    }
}

static inline int roll_die(int faces) {               // [1..faces]
    return (int)uniform_u64_unbiased((uint64_t)faces) + 1;
}

// -------- parse NdM+K ------------------------------------------------------------
static bool parse_spec(const std::string& s, RollSpec& out){
    // formats: "d6", "1d20", "3d6+2", "4d8-1"
    int N=1, M=6, K=0;
    size_t i=0;
    auto read_int = [&](int& v)->bool{
        bool neg=false; if (i<s.size() && (s[i]=='+'||s[i]=='-')) { neg = (s[i]=='-'); ++i; }
        if (i>=s.size() || !std::isdigit((unsigned char)s[i])) return false;
        long val=0; while(i<s.size() && std::isdigit((unsigned char)s[i])) { val = val*10 + (s[i]-'0'); ++i; }
        v = neg? -(int)val : (int)val; return true;
    };
    // optional N
    size_t save=i;
    if (!read_int(N)) { N=1; i=save; }
    if (i>=s.size() || (s[i]!='d' && s[i]!='D')) return false; ++i;
    if (!read_int(M)) return false;
    if (i<s.size()) {
        if (s[i]=='+'){ ++i; if (!read_int(K)) return false; }
        else if (s[i]=='-'){ ++i; int t=0; if(!read_int(t)) return false; K=-t; }
    }
    if (N<1 || M<2) return false;
    out.N=N; out.M=M; out.K=K; return true;
}

// -------- chi-square (p-value) ---------------------------------------------------
static double gammaln(double z){ // Lanczos
    static const double c[6] = {76.18009172947146,-86.50532032941677,24.01409824083091,
                                -1.231739572450155,0.001208650973866179,-0.000005395239384953};
    double x=z, y=z, tmp=x+5.5; tmp -= (x+0.5)*std::log(tmp);
    double ser=1.000000000190015; for(int j=0;j<6;++j){ y+=1.0; ser+=c[j]/y; }
    return -tmp + std::log(2.5066282746310005*ser/x);
}
static double gammap(double s, double x){ // lower regularized P(s,x)
    if (x<=0) return 0.0; const int ITMAX=1000; const double EPS=1e-12;
    double ap=s, sum=1.0/s, del=sum;
    for(int n=1;n<=ITMAX;++n){ ap+=1.0; del*=x/ap; sum+=del; if(std::fabs(del)<std::fabs(sum)*EPS) break; }
    return sum * std::exp(-x + s*std::log(x) - gammaln(s));
}
static double chi_square_pvalue(const std::vector<uint64_t>& counts){
    uint64_t n=0; for(auto c:counts) n+=c; if(n==0) return 1.0;
    size_t k=counts.size(); double expct = (double)n / (double)k;
    double chi2=0.0; for(auto c:counts){ double d=(double)c - expct; chi2 += d*d/expct; }
    double s = 0.5*(double)(k-1); double x = 0.5*chi2;
    double p = 1.0 - gammap(s, x); // upper tail
    return p;
}

// -------- main -------------------------------------------------------------------
int main(int argc, char** argv){
    // Defaults (D6, 100 rolls)
    RollSpec spec{1,6,0};
    int faces = 0, count = 100;
    std::string spec_str;
    std::string primes_path;      // optional
    std::string json_log_path;    // optional per-roll JSON log
    std::string csv_path;         // optional CSV
    bool run_chi=false;
    bool seed_per_roll=true;      // default: new seed for EACH roll from next prime
    bool seed_per_bundle=false;   // alternative: one seed per bundle = per NdM block

    // Parse args (very light)
    for(int i=1;i<argc;++i){
        std::string a = argv[i];
        auto need = [&](const char* flag){ if (i+1>=argc){ std::cerr<<"Missing value for "<<flag<<"\n"; std::exit(2);} return std::string(argv[++i]); };
        if (a=="--faces") { faces = std::stoi(need("--faces")); }
        else if (a=="--count") { count = std::stoi(need("--count")); }
        else if (a=="--spec") { spec_str = need("--spec"); }
        else if (a=="--use-prime-seeds") { primes_path = need("--use-prime-seeds"); }
        else if (a=="--log-json") { json_log_path = need("--log-json"); }
        else if (a=="--csv") { csv_path = need("--csv"); }
        else if (a=="--chi") { run_chi = true; }
        else if (a=="--seed-per-bundle"){ seed_per_bundle=true; seed_per_roll=false; }
        else if (a=="--seed-per-roll"){ seed_per_roll=true; seed_per_bundle=false; }
        else {
            std::cerr << "Unknown arg: " << a << "\n";
            std::cerr << "Usage: --spec NdM+K [--count N] | --faces M --count N [--use-prime-seeds primes.json] [--chi] [--log-json file] [--csv file] [--seed-per-roll|--seed-per-bundle]\n";
            return 2;
        }
    }

    if (!spec_str.empty()) {
        if (!parse_spec(spec_str, spec)) { std::cerr<<"Bad --spec format\n"; return 2; }
        faces = spec.M;
    } else {
        if (faces <= 0) faces = 6;
        spec = RollSpec{1, faces, 0}; // roll one die per trial if --faces/--count mode
    }
    if (faces < 2) { std::cerr<<"faces must be >= 2\n"; return 2; }
    if (count < 1) { std::cerr<<"count must be >= 1\n"; return 2; }

    // Load primes (optional)
    std::vector<uint64_t> primes;
    if (!primes_path.empty()) {
        primes = load_primes_json(primes_path);
        if (primes.empty()) {
            std::cerr << "WARN: no primes loaded; falling back to default seed.\n";
        }
    }
    if (primes.empty()) {
        // Stable default seed so runs are reproducible even without primes
        g_rng.seed(0xA02BDBF7BB3C0A7ULL);
    }

    // Outputs
    std::ofstream json_out, csv_out;
    if (!json_log_path.empty()) { json_out.open(json_log_path); if(!json_out){ std::cerr<<"Cannot open "<<json_log_path<<"\n"; return 2; } json_out << "[\n"; }
    if (!csv_path.empty()) { csv_out.open(csv_path); if(!csv_out){ std::cerr<<"Cannot open "<<csv_path<<"\n"; return 2; } csv_out << "trial,die_result\n"; }

    // Tally for chi-square
    std::vector<uint64_t> counts((size_t)faces, 0);

    auto prime_at = [&](uint64_t i)->uint64_t{
        if (primes.empty()) return 0xDEADBEEFCAFEBABEULL + i*0x9E3779B97F4A7C15ULL;
        return primes[(size_t)(i % primes.size())];
    };

    // Core loop
    uint64_t trial_idx = 0;
    bool first_json = true;
    for (int t=0; t<count; ++t, ++trial_idx) {
        if (!spec_str.empty()) {
            // roll NdM + K as a bundle (optionally bundle-seeded)
            if (seed_per_bundle && !primes.empty()) seed_from_prime(prime_at(trial_idx));
            int total = spec.K;
            std::vector<int> each; each.reserve(spec.N);
            for (int d=0; d<spec.N; ++d) {
                if (seed_per_roll && !primes.empty()) seed_from_prime(prime_at(trial_idx* (uint64_t)spec.N + d));
                int r = roll_die(spec.M);
                each.push_back(r);
                counts[(size_t)(r-1)]++;
                total += r;
            }
            // Logs
            if (json_out) {
                if (!first_json) json_out << ",\n";
                first_json = false;
                json_out << "  {\"trial\":"<<t<<",\"spec\":\""<<spec_str<<"\",\"rolls\":[";
                for (size_t i=0;i<each.size();++i){ if(i) json_out<<","; json_out<<each[i]; }
                json_out << "],\"modifier\":"<<spec.K<<",\"total\":"<<total<<"}";
            }
            if (csv_out) {
                for (size_t i=0;i<each.size();++i) csv_out << t << "," << each[i] << "\n";
            }
            // Also print to stdout (concise)
            std::cout << total << (t+1<count? ' ' : '\n');
        } else {
            // simple single-die per trial
            if (seed_per_roll && !primes.empty()) seed_from_prime(prime_at(trial_idx));
            int r = roll_die(faces);
            counts[(size_t)(r-1)]++;
            if (json_out) {
                if (!first_json) json_out << ",\n";
                first_json = false;
                json_out << "  {\"trial\":"<<t<<",\"faces\":"<<faces<<",\"roll\":"<<r<<"}";
            }
            if (csv_out) csv_out << t << "," << r << "\n";
            std::cout << r << (t+1<count? ' ' : '\n');
        }
    }

    if (json_out) { json_out << "\n]\n"; json_out.close(); }
    if (csv_out) csv_out.close();

    if (run_chi) {
        double p = chi_square_pvalue(counts);
        std::cerr << "[chi] faces="<<faces<<"  rolls="<<count<<"  p="<<p<<"\n";
        std::cerr << "Counts:"; for(size_t i=0;i<counts.size();++i) std::cerr << " " << (i+1) << ":" << counts[i]; std::cerr << "\n";
    }
    return 0;
}