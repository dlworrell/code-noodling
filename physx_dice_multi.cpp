// physx_dice_multi.cpp
// Auto-tuned PhysX dice simulator with NdM+K parsing, multi-scene parallelism,
// prime-seeded determinism, JSON/CSV logs, chi-square, and full physical support
// for D6/D8/D12/D20 via convex meshes + local-face-normal top-face detection.
//
// Build notes (CMake):
//   add_executable(physx_dice_multi physx_dice_multi.cpp)
//   target_compile_options(physx_dice_multi PRIVATE -O3 -march=native)
//   find_library(PHYSX_LIB PhysX_64)
//   find_library(PHYSX_FOUNDATION_LIB PhysXFoundation_64)
//   find_library(PHYSX_COMMON_LIB PhysXCommon_64)
//   find_library(PHYSX_EXT_LIB PhysXExtensions_static)
//   find_library(PHYSX_COOKING_LIB PhysXCooking_64)
//   target_link_libraries(physx_dice_multi PRIVATE
//       ${PHYSX_LIB} ${PHYSX_FOUNDATION_LIB} ${PHYSX_COMMON_LIB}
//       ${PHYSX_EXT_LIB} ${PHYSX_COOKING_LIB})
//
// Usage examples:
//   ./physx_dice_multi --spec 1d6 --trials 50000 --chi
//   ./physx_dice_multi --spec 3d6+2 --spec 1d8 --spec 1d12 --spec 1d20 \
//       --trials 20000 --use-prime-seeds primes.json --json runs.json --csv counts.csv --chi
//
// Requires: die_mesh.h (in the same directory)

#include <PxPhysicsAPI.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "die_mesh.h"

using namespace physx;

/* ------------------------- Tiny JSON primes loader ------------------------- */
static std::vector<uint64_t> load_primes_json(const std::string& path){
    std::ifstream in(path);
    if(!in){ std::cerr<<"WARN: cannot open primes: "<<path<<"\n"; return {}; }
    std::string s((std::istreambuf_iterator<char>(in)), {});
    std::vector<uint64_t> p; p.reserve(1000);
    auto pos = s.find("\"primes\"");
    if(pos==std::string::npos){ std::cerr<<"WARN: primes key not found\n"; return p; }
    pos = s.find('[', pos); if(pos==std::string::npos) return p;
    auto end = s.find(']', pos); if(end==std::string::npos) return p;
    std::string arr = s.substr(pos+1, end-pos-1);
    size_t i=0;
    while(i<arr.size()){
        while(i<arr.size() && (arr[i]==' '||arr[i]==','||arr[i]=='\n'||arr[i]=='\r'||arr[i]=='\t')) ++i;
        size_t j=i;
        while(j<arr.size() && std::isdigit((unsigned char)arr[j])) ++j;
        if(j>i) p.push_back(std::stoull(arr.substr(i, j-i)));
        i = (j==i? i+1 : j);
    }
    return p;
}

/* ------------------------- RNG & unbiased sampler ------------------------- */
static thread_local std::mt19937_64 g_rng{0xA02BDBF7BB3C0A7ULL};
static inline uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline void seed_from_prime(uint64_t p){ g_rng.seed(splitmix64(p)); }

// Lemire + rejection: unbiased [0..n-1] for any n
static inline uint64_t uniform_u64_unbiased(uint64_t n){
    if (n == 0) return 0;
    for(;;){
        uint64_t x = g_rng();
        __uint128_t m = (__uint128_t)x * (__uint128_t)n;
        uint64_t l = (uint64_t)m;
        if (l < n) {
            uint64_t t = (-n) % n;
            if (l < t) continue;
        }
        return (uint64_t)(m >> 64);
    }
}
static inline int roll_unbiased(int faces){ return (int)uniform_u64_unbiased((uint64_t)faces) + 1; }

/* ------------------------- Chi-square p-value ------------------------------ */
static double gammaln(double z){
    static const double c[6] = {76.18009172947146,-86.50532032941677,24.01409824083091,
                                -1.231739572450155,0.001208650973866179,-0.000005395239384953};
    double x=z, y=z, tmp=x+5.5; tmp -= (x+0.5)*std::log(tmp);
    double ser=1.000000000190015; for(int j=0;j<6;++j){ y+=1.0; ser+=c[j]/y; }
    return -tmp + std::log(2.5066282746310005*ser/x);
}
static double gammap(double s, double x){
    if (x<=0) return 0.0; const int ITMAX=1000; const double EPS=1e-12;
    double ap=s, sum=1.0/s, del=sum;
    for(int n=1;n<=ITMAX;++n){ ap+=1.0; del*=x/ap; sum+=del; if(std::fabs(del)<std::fabs(sum)*EPS) break; }
    return sum * std::exp(-x + s*std::log(x) - gammaln(s));
}
static double chi_square_pvalue(const std::vector<uint64_t>& counts){
    uint64_t n=0; for(auto c:counts) n+=c; if(n==0) return 1.0;
    size_t k=counts.size(); double expct = (double)n / (double)k;
    double chi2=0.0; for(auto c:counts){ double d=(double)c - expct; chi2 += d*d/expct; }
    return 1.0 - gammap(0.5*(k-1), 0.5*chi2);
}

/* ------------------------- Dice spec parsing ------------------------------- */
struct Spec { int N=1, M=6, K=0; std::string label; };
static bool parse_spec(const std::string& s, Spec& out){
    // supports "d6","1d20","3d6+2","4d8-1"
    int N=1,M=6,K=0; size_t i=0, n=s.size();
    auto read_int=[&](int& v)->bool{
        bool neg=false; if(i<n && (s[i]=='+'||s[i]=='-')){neg=(s[i]=='-');++i;}
        if(i>=n || !std::isdigit((unsigned char)s[i])) return false;
        long val=0; while(i<n && std::isdigit((unsigned char)s[i])){ val=val*10+(s[i]-'0'); ++i; }
        v = neg? -(int)val : (int)val; return true;
    };
    size_t save=i; if(!read_int(N)){ N=1; i=save; }
    if(i>=n || (s[i]!='d'&&s[i]!='D')) return false; ++i;
    if(!read_int(M)) return false;
    if(i<n){ if(s[i]=='+'){++i; if(!read_int(K)) return false;}
             else if(s[i]=='-'){++i; int t=0; if(!read_int(t)) return false; K=-t;} }
    if(N<1||M<2) return false;
    out.N=N; out.M=M; out.K=K; out.label=s; return true;
}

/* ------------------------- PhysX helpers ----------------------------------- */
static PxDefaultAllocator      gAlloc;
static PxDefaultErrorCallback  gErr;

struct PxStuff {
    PxFoundation*  fnd=nullptr;
    PxPhysics*     phy=nullptr;
};
static PxStuff makePx(){
    PxStuff s;
    s.fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    s.phy = PxCreatePhysics(PX_PHYSICS_VERSION, *s.fnd, PxTolerancesScale());
    return s;
}
static void freePx(PxStuff& s){
    if(s.phy){ s.phy->release(); s.phy=nullptr; }
    if(s.fnd){ s.fnd->release(); s.fnd=nullptr; }
}
struct ScenePack {
    PxScene*    scene=nullptr;
    PxMaterial* mat=nullptr;
};
static ScenePack makeScene(PxPhysics* phy, int cpuThreads){
    ScenePack sp;
    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0,-9.81f,0);
    sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(std::max(1,cpuThreads));
    sd.filterShader = PxDefaultSimulationFilterShader;
    sp.scene = phy->createScene(sd);
    sp.mat   = phy->createMaterial(0.6f,0.6f,0.25f);
    // ground
    PxRigidStatic* plane = PxCreatePlane(*phy, PxPlane(0,1,0,0), *sp.mat);
    sp.scene->addActor(*plane);
    return sp;
}
static void freeScene(ScenePack& sp){
    if(sp.scene){ sp.scene->release(); sp.scene=nullptr; }
    if(sp.mat){ sp.mat->release(); sp.mat=nullptr; }
}

static void stepUntilSettle(PxScene* scn, PxRigidDynamic* die, int maxSteps=2000, int settleFrames=40){
    int stable=0;
    for(int i=0;i<maxSteps;++i){
        scn->simulate(1.0f/120.0f); scn->fetchResults(true);
        PxVec3 lv=die->getLinearVelocity(), av=die->getAngularVelocity();
        if (lv.magnitude()<0.05f && av.magnitude()<0.05f) ++stable; else stable=0;
        if (stable>=settleFrames) break;
    }
}

/* ------------------------- Worker (one thread) ------------------------------ */
struct Task { Spec spec; uint64_t trials=0; };
struct SpecAgg {
    Spec spec;
    std::vector<uint64_t> counts; // per-face counts
    std::atomic<uint64_t> total_trials{0};
    std::mutex m;
    SpecAgg(){} SpecAgg(const Spec& s):spec(s),counts((size_t)s.M,0){}
};

static void run_worker(Task t, SpecAgg* agg,
                       const std::vector<uint64_t>* primes,
                       bool seed_per_roll,
                       int cpuThreadsInScene = 2)
{
    // local RNG (prime-seeded as needed)
    std::mt19937_64 rng{0xA02BDBF7BB3C0A7ULL};
    auto prime_at=[&](uint64_t i)->uint64_t{
        if (!primes || primes->empty()) return 0xDEADBEEFCAFEBABEULL + i*0x9E3779B97F4A7C15ULL;
        return (*primes)[(size_t)(i % primes->size())];
    };

    // Will we simulate physically?
    bool doPhys = (t.spec.M==6 || t.spec.M==8 || t.spec.M==12 || t.spec.M==20);

    // Build per-thread PhysX context only if needed
    PxStuff px; ScenePack sp;
    PxCooking* cooking = nullptr;
    DieMesh dm;
    if (doPhys){
        px = makePx();
        sp = makeScene(px.phy, cpuThreadsInScene);

        // Optional cooking for convexes
        PxCookingParams cp(px.phy->getTolerancesScale());
        cooking = PxCreateCooking(PX_PHYSICS_VERSION, *px.fnd, cp);

        // Choose and build die mesh
        if (t.spec.M==6) {
            dm = make_d6_box();
        } else if (t.spec.M==8) {
            dm = make_d8(*px.phy, 1.0f);
            std::vector<PxVec3> octPts = { {+1,0,0},{-1,0,0},{0,+1,0},{0,-1,0},{0,0,+1},{0,0,-1} };
            dm.convex = cook_points_to_convex(*px.phy, *cooking, octPts, 1.0f);
        } else if (t.spec.M==12) {
            dm = make_d12(*px.phy, 1.0f);
            std::vector<PxVec3> pts = {
              {+1,+1,+1},{+1,+1,-1},{+1,-1,+1},{+1,-1,-1},
              {-1,+1,+1},{-1,+1,-1},{-1,-1,+1},{-1,-1,-1},
              {0, 1.618f, 0},{0,-1.618f,0},{1.618f,0,0},{-1.618f,0,0},{0,0,1.618f},{0,0,-1.618f}
            };
            dm.convex = cook_points_to_convex(*px.phy, *cooking, pts, 1.0f);
        } else if (t.spec.M==20) {
            dm = make_d20(*px.phy, 1.0f);
            float phi = (1.0f + std::sqrt(5.0f))*0.5f;
            std::vector<PxVec3> pts = {
                {-1,  phi,  0},{ 1,  phi,  0},{ -1, -phi,  0},{ 1, -phi,  0},
                { 0, -1,  phi},{ 0, 1,  phi},{  0, -1, -phi},{ 0, 1, -phi},
                { phi, 0, -1},{ phi, 0, 1},{ -phi, 0, -1},{ -phi,0, 1}
            };
            dm.convex = cook_points_to_convex(*px.phy, *cooking, pts, 1.0f);
        }
    }

    std::vector<uint64_t> local_counts((size_t)t.spec.M, 0);

    for (uint64_t trial=0; trial < t.trials; ++trial){
        if (seed_per_roll) rng.seed(splitmix64(prime_at(trial)));

        int sum = t.spec.K;
        if (doPhys){
            for (int i=0;i<t.spec.N;++i){
                PxRigidDynamic* die = spawn_die_mesh(px.phy, sp.scene, sp.mat, dm, PxVec3(0,3.5f,0), rng);
                stepUntilSettle(sp.scene, die);
                int face = top_face_from_normals(die->getGlobalPose().q, dm.faceNormalsLocal, dm.faceNumber);
                local_counts[(size_t)(face-1)]++;
                sum += face;
                sp.scene->removeActor(*die); die->release();
            }
        } else {
            // Unbiased virtual for other M
            for (int i=0;i<t.spec.N;++i){
                int face = (int)uniform_u64_unbiased((uint64_t)t.spec.M) + 1;
                local_counts[(size_t)(face-1)]++;
                sum += face;
            }
        }
        (void)sum; // totals not aggregated globally in this version
    }

    // merge into aggregator
    {
        std::lock_guard<std::mutex> lk(agg->m);
        for (size_t i=0;i<local_counts.size();++i) agg->counts[i] += local_counts[i];
        agg->total_trials += t.trials;
    }

    if (doPhys){
        if (cooking){ cooking->release(); cooking=nullptr; }
        freeScene(sp);
        freePx(px);
    }
}

/* ------------------------- Main -------------------------------------------- */
int main(int argc, char** argv){
    // CLI
    std::vector<Spec> specs;   // multiple allowed
    uint64_t trials = 10000;   // per-spec trials
    std::string primes_path, json_path, csv_path;
    bool chi=false, seed_per_roll=true, interleave=false;

    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto need=[&](const char* f){ if(i+1>=argc){ std::cerr<<"Missing value for "<<f<<"\n"; std::exit(2);} return std::string(argv[++i]); };
        if (a=="--spec"){ Spec s; if(!parse_spec(need("--spec"), s)){ std::cerr<<"Bad spec\n"; return 2; } specs.push_back(s); }
        else if (a=="--trials"){ trials = std::stoull(need("--trials")); }
        else if (a=="--use-prime-seeds"){ primes_path = need("--use-prime-seeds"); }
        else if (a=="--json"){ json_path = need("--json"); }
        else if (a=="--csv"){ csv_path = need("--csv"); }
        else if (a=="--chi"){ chi=true; }
        else if (a=="--seed-per-roll"){ seed_per_roll=true; }
        else if (a=="--seed-per-bundle"){ seed_per_roll=false; } // (future enhancement)
        else if (a=="--mix"){ interleave=true; } // round-robin specs
        else { std::cerr<<"Unknown arg: "<<a<<"\n"; return 2; }
    }
    if (specs.empty()) { Spec s; s.label="1d6"; specs.push_back(s); }
    if (trials<1) { std::cerr<<"trials must be >=1\n"; return 2; }

    // Detect hardware threads; choose worker threads & scene threads
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    // Use ~75% of cores for workers (leave some for OS), min 1, max 16
    unsigned workers = std::clamp((unsigned)std::max(1u, (unsigned)(hw*3/4)), 1u, 16u);
    // Per-scene PhysX dispatcher threads (1-3 is usually best)
    int sceneThreads = (hw >= 8 ? 3 : (hw >= 4 ? 2 : 1));

    // Load primes (optional)
    std::vector<uint64_t> primes;
    if (!primes_path.empty()){
        primes = load_primes_json(primes_path);
        if (primes.empty()) std::cerr<<"WARN: no primes loaded; using fixed seed.\n";
    }
    if (primes.empty()) g_rng.seed(0xA02BDBF7BB3C0A7ULL);

    // Prepare aggregators (one per spec)
    std::vector<SpecAgg> aggs; aggs.reserve(specs.size());
    for (auto& s: specs) aggs.emplace_back(s);

    // Create tasks
    std::vector<Task> tasks;
    if (!interleave){
        for (auto& s: specs){ Task t; t.spec=s; t.trials=trials; tasks.push_back(t); }
    }else{
        uint64_t each = trials;
        for (auto& s: specs){ Task t; t.spec=s; t.trials=each; tasks.push_back(t); }
    }

    // Run workers
    std::vector<std::future<void>> futs;
    futs.reserve(workers);
    std::atomic<size_t> task_idx{0};
    auto worker_fn = [&](){
        for(;;){
            size_t idx = task_idx.fetch_add(1);
            if (idx >= tasks.size()) break;
            // Split each task into shards for better parallelism
            Task base = tasks[idx];
            uint64_t shard = std::max<uint64_t>(100, base.trials / workers);
            uint64_t done=0;
            while (done < base.trials){
                Task t = base; t.trials = std::min<uint64_t>(shard, base.trials - done);
                run_worker(t, &aggs[idx], primes.empty()? nullptr : &primes, seed_per_roll, sceneThreads);
                done += t.trials;
            }
        }
    };
    auto t0 = std::chrono::high_resolution_clock::now();
    for (unsigned w=0; w<workers; ++w) futs.emplace_back(std::async(std::launch::async, worker_fn));
    for (auto& f: futs) f.get();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    // Logs
    std::ofstream jout, coutf;
    bool jfirst=true;
    if (!json_path.empty()){ jout.open(json_path); if(!jout){ std::cerr<<"Cannot open "<<json_path<<"\n"; return 2; } jout<<"{\n  \"runs\": [\n"; }
    if (!csv_path.empty()){ coutf.open(csv_path); if(!coutf){ std::cerr<<"Cannot open "<<csv_path<<"\n"; return 2; } coutf<<"spec,face,count\n"; }

    for (size_t si=0; si<specs.size(); ++si){
        auto& s = specs[si];
        auto& A = aggs[si];
        if (csv_path.size()){
            for (size_t f=0; f<A.counts.size(); ++f){
                coutf<<s.label<<","<<(f+1)<<","<<A.counts[f]<<"\n";
            }
        }
        if (json_path.size()){
            if(!jfirst) jout<<",\n"; jfirst=false;
            jout<<"    {\"spec\":\""<<s.label<<"\",\"faces\":"<<s.M<<",\"trials\":"<<A.total_trials<<",\"counts\":[";
            for(size_t f=0; f<A.counts.size(); ++f){
                if(f) jout<<","; jout<<A.counts[f];
            }
            jout<<"]}";
        }
        if (chi){
            double p = chi_square_pvalue(A.counts);
            std::cerr<<"[chi] "<<s.label<<"  trials="<<A.total_trials<<"  p="<<p<<"\n";
            std::cerr<<"Counts:"; for(size_t f=0; f<A.counts.size(); ++f) std::cerr<<" "<<(f+1)<<":"<<A.counts[f]; std::cerr<<"\n";
        }
    }
    if (json_path.size()){
        jout<<"\n  ],\n  \"elapsed_ms\": "<<ms<<", \"workers\": "<<workers<<", \"scene_threads\": "<<sceneThreads<<"\n}\n";
    }

    std::cerr<<"[physx_dice_multi] specs="<<specs.size()<<" workers="<<workers
             <<" sceneThreads="<<sceneThreads<<" time="<<ms<<" ms\n";
    return 0;
}