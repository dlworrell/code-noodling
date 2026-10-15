// physx_dice.cpp — PhysX-based, prime-seeded D6 simulator with JSON/CSV and chi-square
// Build (CMake target example at the end):
//   add_executable(physx_dice physx_dice.cpp)
//   target_link_libraries(physx_dice PRIVATE PhysX_64 PhysXFoundation_64 PhysXCommon_64 PhysXExtensions_static)
//   target_compile_definitions(physx_dice PRIVATE _CRT_SECURE_NO_WARNINGS)
//
// Usage examples:
//   ./physx_dice --rolls 5000 --chi
//   ./physx_dice --rolls 5000 --use-prime-seeds primes.json --json rolls.json --chi
//   ./physx_dice --rolls 2000 --csv rolls.csv --json rolls.json
//
// Notes:
//  - This uses a BOX (cube) for a D6 with edge=1.0 (half-extents 0.5).
//  - “Top face” is derived from actor quaternion by checking which local face normal
//    aligns best with world +Y (up). Mapping is configurable below.
//  - To support D8/D12/D20 later, replace the shape with a convex mesh and provide
//    a table of local-face normals -> face numbers, then reuse the same top-face logic.

#include <PxPhysicsAPI.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cctype>
#include <cmath>
#include <cstdio>

using namespace physx;

// ---------- tiny JSON loader for ["primes":[...]] ----------
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

// ---------- RNG + seeding ----------
static thread_local std::mt19937_64 g_rng{0xA02BDBF7BB3C0A7ULL};
static inline uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline void seed_from_prime(uint64_t p){ g_rng.seed(splitmix64(p)); }

// ---------- chi-square (p-value) ----------
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
    double s = 0.5*(double)(k-1), x = 0.5*chi2;
    return 1.0 - gammap(s, x);
}

// ---------- PhysX globals ----------
static PxDefaultAllocator      gAllocator;
static PxDefaultErrorCallback  gErrorCb;
static PxFoundation*           gFoundation = nullptr;
static PxPhysics*              gPhysics    = nullptr;
static PxScene*                gScene      = nullptr;
static PxMaterial*             gMaterial   = nullptr;

// Optional: create a ground plane
static PxRigidStatic* createPlane(PxPhysics& physics){
    PxRigidStatic* plane = PxCreatePlane(physics, PxPlane(0,1,0,0), *gMaterial);
    gScene->addActor(*plane);
    return plane;
}

// Init & cleanup
static void initPhysX(){
    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCb);
    gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale());
    PxSceneDesc sd(gPhysics->getTolerancesScale());
    sd.gravity = PxVec3(0.0f,-9.81f,0.0f);
    sd.cpuDispatcher = PxDefaultCpuDispatcherCreate(2);
    sd.filterShader = PxDefaultSimulationFilterShader;
    gScene = gPhysics->createScene(sd);
    gMaterial = gPhysics->createMaterial(0.6f, 0.6f, 0.25f); // staticF, dynamicF, restitution
    createPlane(*gPhysics);
}
static void cleanupPhysX(){
    if(gScene){ gScene->release(); gScene=nullptr; }
    if(gPhysics){ gPhysics->release(); gPhysics=nullptr; }
    if(gFoundation){ gFoundation->release(); gFoundation=nullptr; }
}

// Create one D6 (cube) with impulses from RNG
static PxRigidDynamic* spawnD6(const PxVec3& pos){
    const float he = 0.5f; // half-extent (edge length = 1.0)
    PxRigidDynamic* die = PxCreateDynamic(*gPhysics, PxTransform(pos),
                                          PxBoxGeometry(he,he,he), *gMaterial, 1.0f);
    die->setAngularDamping(0.05f);
    die->setLinearDamping(0.01f);
    die->setSleepThreshold(0.05f);
    die->setName("D6");

    // Prime-seeded RNG drives impulses deterministically
    std::uniform_real_distribution<float> Upos(-0.25f, 0.25f);
    std::uniform_real_distribution<float> Uimp( 2.0f, 8.0f);
    std::uniform_real_distribution<float> Utor( 1.0f, 6.0f);

    // Small random offset within a funnel above the plane
    PxTransform t = die->getGlobalPose();
    t.p.x += Upos(g_rng);
    t.p.z += Upos(g_rng);
    die->setGlobalPose(t);

    // Linear impulse (random direction, upward bias)
    PxVec3 imp( Uimp(g_rng)*(float)((g_rng()&1)? 1:-1),
                Uimp(g_rng),
                Uimp(g_rng)*(float)((g_rng()&1)? 1:-1) );
    die->addForce(imp, PxForceMode::eIMPULSE);

    // Angular impulse
    PxVec3 tor( Utor(g_rng)*(float)((g_rng()&1)? 1:-1),
                Utor(g_rng)*(float)((g_rng()&1)? 1:-1),
                Utor(g_rng)*(float)((g_rng()&1)? 1:-1) );
    die->addTorque(tor, PxForceMode::eIMPULSE);

    gScene->addActor(*die);
    return die;
}

// Step simulation until settle or timeout
static void stepSimUntilSettle(PxRigidDynamic* die, int maxSteps=2000, int settleFrames=30){
    int stable=0;
    for(int i=0;i<maxSteps;++i){
        gScene->simulate(1.0f/120.0f);  // 120 Hz for better stability
        gScene->fetchResults(true);
        // check velocities
        PxVec3 lv = die->getLinearVelocity();
        PxVec3 av = die->getAngularVelocity();
        float ls = lv.magnitude(), as = av.magnitude();
        if (ls < 0.05f && as < 0.05f) ++stable; else stable=0;
        if (stable >= settleFrames) break;
    }
}

// Map top face from quaternion: pick local face normal with max dot to world +Y.
// Configure which numbers correspond to local +X,-X,+Y,-Y,+Z,-Z.
static int faceFromQuat_D6(const PxQuat& q){
    // Convert local axes to world
    auto rotate = [&](const PxVec3& v){ return q.rotate(v); };
    PxVec3 nx = rotate(PxVec3( 1,0,0));
    PxVec3 px = rotate(PxVec3(-1,0,0));
    PxVec3 ny = rotate(PxVec3( 0,1,0));
    PxVec3 py = rotate(PxVec3( 0,-1,0));
    PxVec3 nz = rotate(PxVec3( 0,0,1));
    PxVec3 pz = rotate(PxVec3( 0,0,-1));

    // Dot with world up
    const PxVec3 up(0,1,0);
    float d_nx = nx.dot(up);
    float d_px = px.dot(up);
    float d_ny = ny.dot(up);
    float d_py = py.dot(up);
    float d_nz = nz.dot(up);
    float d_pz = pz.dot(up);

    // Choose the most upward-facing normal
    float dmax = d_nx; int face = 0;
    auto upd = [&](float d, int f){ if (d>dmax){ dmax=d; face=f; } };
    // Face numbering mapping (edit to match your physical pips/orientation)
    // Suppose:
    //   local +Y -> face 1
    //   local -Y -> face 6
    //   local +X -> face 2
    //   local -X -> face 5
    //   local +Z -> face 3
    //   local -Z -> face 4
    upd(d_ny, 1); // +Y
    upd(d_py, 6); // -Y
    upd(d_nx, 2); // +X
    upd(d_px, 5); // -X
    upd(d_nz, 3); // +Z
    upd(d_pz, 4); // -Z

    return face;
}

int main(int argc, char** argv){
    // ---- CLI ----
    int rolls = 1000;
    std::string primes_path;
    std::string json_path;
    std::string csv_path;
    bool run_chi=false;

    for(int i=1;i<argc;++i){
        std::string a = argv[i];
        auto need=[&](const char* f){ if(i+1>=argc){ std::cerr<<"Missing value for "<<f<<"\n"; std::exit(2);} return std::string(argv[++i]); };
        if (a=="--rolls") rolls = std::stoi(need("--rolls"));
        else if (a=="--use-prime-seeds") primes_path = need("--use-prime-seeds");
        else if (a=="--json") json_path = need("--json");
        else if (a=="--csv") csv_path = need("--csv");
        else if (a=="--chi") run_chi = true;
        else { std::cerr<<"Unknown arg: "<<a<<"\n"; return 2; }
    }
    if (rolls < 1) { std::cerr<<"rolls must be >=1\n"; return 2; }

    // ---- Seeds from primes (optional) ----
    std::vector<uint64_t> primes;
    if (!primes_path.empty()){
        primes = load_primes_json(primes_path);
        if (primes.empty())
            std::cerr<<"WARN: no primes loaded; using fixed seed.\n";
    }
    if (primes.empty()){
        g_rng.seed(0xA02BDBF7BB3C0A7ULL);
    }

    // ---- Outputs ----
    std::ofstream jout, coutf;
    bool jfirst=true;
    if (!json_path.empty()){ jout.open(json_path); if(!jout){ std::cerr<<"Cannot open "<<json_path<<"\n"; return 2; } jout<<"[\n"; }
    if (!csv_path.empty()){ coutf.open(csv_path); if(!coutf){ std::cerr<<"Cannot open "<<csv_path<<"\n"; return 2; } coutf<<"trial,face\n"; }

    // ---- PhysX world ----
    initPhysX();

    // ---- Roll loop ----
    std::vector<uint64_t> counts(6,0);
    for(int t=0;t<rolls;++t){
        // per-roll deterministic seed
        if(!primes.empty()) seed_from_prime(primes[(size_t)(t % primes.size())]);

        // Spawn & simulate
        PxRigidDynamic* die = spawnD6(PxVec3(0, 3.5f, 0));
        stepSimUntilSettle(die, /*maxSteps*/2000, /*settleFrames*/40);

        // Top face
        PxTransform pose = die->getGlobalPose();
        int face = faceFromQuat_D6(pose.q);
        counts[(size_t)(face-1)]++;

        // Logs
        if (jout){
            if(!jfirst) jout<<",\n"; jfirst=false;
            jout<<"  {\"trial\":"<<t<<",\"face\":"<<face<<"}";
        }
        if (coutf) coutf<<t<<","<<face<<"\n";

        // Destroy the die
        gScene->removeActor(*die);
        die->release();
    }

    if (jout){ jout<<"\n]\n"; jout.close(); }
    if (coutf) coutf.close();

    if (run_chi){
        double p = chi_square_pvalue(counts);
        std::cerr<<"[chi] D6  rolls="<<rolls<<"  p="<<p<<"\n";
        std::cerr<<"Counts:"; for(size_t i=0;i<counts.size();++i) std::cerr<<" "<<(i+1)<<":"<<counts[i]; std::cerr<<"\n";
    }

    cleanupPhysX();
    return 0;
}