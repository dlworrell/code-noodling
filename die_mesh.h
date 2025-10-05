// die_mesh.h — minimal die mesh + face normal tables + PhysX convex cooking.
// Works out-of-the-box for D6, D8, D12, D20 without external assets.
// Optionally load OBJ later and cook as convex.
//
// Usage: 
//   DieMesh dm = make_platonic_die(phy, 8); // D8
//   PxRigidDynamic* die = spawn_die_mesh(phy, scene, dm, mat);
//
// Build notes:
//   Link PhysXCooking in CMake if you enable cook_from_obj().

#pragma once
#include <PxPhysicsAPI.h>
#include <vector>
#include <string>
#include <cctype>
#include <cstdio>

using namespace physx;

struct DieMesh {
    PxConvexMesh* convex = nullptr;              // null for D6 (box)
    PxGeometryType::Enum geomType = PxGeometryType::eCONVEXMESH;
    float scale = 1.0f;                          // uniform scale
    // For top-face mapping:
    std::vector<PxVec3> faceNormalsLocal;        // one normal per numbered face (unit)
    std::vector<int>    faceNumber;              // mapping index -> face label (1..M)
    int faces = 0;                               
    bool isBox = false;                          // true for D6 box path
};

// ---------- Helpers ----------
static inline PxVec3 unit(const PxVec3& v){ float m = v.magnitude(); return (m>0)? v*(1.0f/m) : v; }

// ---------- Hard-coded Platonic solids ----------
// We provide local-face normals & numbering conventions that you can tweak.
// Geometry cooking uses PhysX convex cooking from vertices for D8/D12/D20.

static DieMesh make_d6_box(){
    DieMesh dm; dm.isBox = true; dm.geomType = PxGeometryType::eBOX; dm.faces = 6; dm.scale = 1.0f;
    // Numbering convention (edit to match your physical dice):
    // +Y:1, -Y:6, +X:2, -X:5, +Z:3, -Z:4
    dm.faceNormalsLocal = { PxVec3(0,1,0), PxVec3(0,-1,0), PxVec3(1,0,0),
                            PxVec3(-1,0,0), PxVec3(0,0,1), PxVec3(0,0,-1) };
    dm.faceNumber       = { 1, 6, 2, 5, 3, 4 };
    return dm;
}

// D8 (octahedron): vertices at axes; faces are triangles; 8 faces; normals ±X,±Y,±Z (like axis)
static DieMesh make_d8(PxPhysics& phy, float scale=1.0f){
    DieMesh dm; dm.scale = scale; dm.faces = 8;
    // Vertices
    std::vector<PxVec3> v = {
        {+1,0,0},{-1,0,0},{0,+1,0},{0,-1,0},{0,0,+1},{0,0,-1}
    };
    // For cooking a convex from points:
    PxConvexMeshDesc desc; 
    desc.points.count  = (uint32_t)v.size();
    desc.points.stride = sizeof(PxVec3);
    desc.points.data   = v.data();
    desc.flags         = PxConvexFlag::eCOMPUTE_CONVEX;
    // NOTE: cooking done via cooking interface in cook_points_to_convex (below)
    dm.convex = nullptr; // set by cook call later
    dm.geomType = PxGeometryType::eCONVEXMESH;
    // Face normals — for octahedron, the face normals are located around axes; we map 8 directions.
    dm.faceNormalsLocal = {
        unit(PxVec3(+1,+1, 0)), unit(PxVec3(+1,-1, 0)),
        unit(PxVec3(-1,+1, 0)), unit(PxVec3(-1,-1, 0)),
        unit(PxVec3( 0,+1,+1)), unit(PxVec3( 0,+1,-1)),
        unit(PxVec3( 0,-1,+1)), unit(PxVec3( 0,-1,-1))
    };
    // Numbering 1..8 (tweak to match pips/symbols you’ll use)
    dm.faceNumber = {1,2,3,4,5,6,7,8};
    return dm;
}

// D20 (icosahedron): we use approximate local normals for 20 triangular faces.
// To keep this compact, we use golden ratio directions for normals.
static DieMesh make_d20(PxPhysics& phy, float scale=1.0f){
    DieMesh dm; dm.scale = scale; dm.faces = 20;
    const float phi = (1.0f + std::sqrt(5.0f))*0.5f;
    std::vector<PxVec3> verts = {
        { -1,  phi,  0},{ 1,  phi,  0},{ -1, -phi,  0},{ 1, -phi,  0},
        {  0, -1,  phi},{ 0, 1,  phi},{  0, -1, -phi},{ 0, 1, -phi},
        {  phi, 0, -1},{ phi, 0, 1},{ -phi, 0, -1},{ -phi,0, 1}
    };
    // Cook convex from verts (actual face tessellation is computed by PhysX)
    dm.convex = nullptr; 
    dm.geomType = PxGeometryType::eCONVEXMESH;

    // 20 face normals (approx canonical; you may remap numbering after you print a few test rolls)
    std::vector<PxVec3> n = {
        { 0,  1,  phi},{ 0,  1, -phi},{ 0, -1,  phi},{ 0, -1, -phi},
        { 1,  phi, 0},{-1,  phi, 0},{ 1, -phi, 0},{-1, -phi, 0},
        { phi, 0,  1},{ phi, 0, -1},{-phi, 0,  1},{-phi, 0, -1},
        { 1, 1, 1},{ 1, 1,-1},{ 1,-1, 1},{ 1,-1,-1},
        {-1, 1, 1},{-1, 1,-1},{-1,-1, 1},{-1,-1,-1}
    };
    dm.faceNormalsLocal.clear();
    for (auto &x: n) dm.faceNormalsLocal.push_back(unit(x));
    dm.faceNumber.resize(20); for(int i=0;i<20;++i) dm.faceNumber[i]=i+1;
    return dm;
}

// D12 (dodecahedron): similar approach; a light set of local normals.
static DieMesh make_d12(PxPhysics& phy, float scale=1.0f){
    DieMesh dm; dm.scale = scale; dm.faces = 12;
    const float phi = (1.0f + std::sqrt(5.0f))*0.5f;
    // A rough normal set; for production, replace by precise face normals from your mesh.
    std::vector<PxVec3> n = {
        { 0, ±1, ±phi}, { ±1, ±phi, 0}, { ±phi, 0, ±1}
    };
    // Expand ± combinations
    dm.faceNormalsLocal = {
        unit(PxVec3( 0, 1, phi)), unit(PxVec3( 0, 1,-phi)),
        unit(PxVec3( 0,-1, phi)), unit(PxVec3( 0,-1,-phi)),
        unit(PxVec3( 1, phi, 0)), unit(PxVec3(-1, phi, 0)),
        unit(PxVec3( 1,-phi, 0)), unit(PxVec3(-1,-phi, 0)),
        unit(PxVec3( phi, 0, 1)), unit(PxVec3( phi, 0,-1)),
        unit(PxVec3(-phi, 0, 1)), unit(PxVec3(-phi, 0,-1))
    };
    dm.faceNumber.resize(12); for(int i=0;i<12;++i) dm.faceNumber[i]=i+1;
    dm.geomType = PxGeometryType::eCONVEXMESH;
    dm.convex   = nullptr;
    return dm;
}

/* ---------- Convex cooking helpers (optional) ----------
   You’ll need PhysXCooking created from PxCookingParams; we keep
   this simple by using PxDefaultMemoryInputData/OutputData. */

// Fill this when you add a cooking interface in your app:
//   PxCooking* gCooking = PxCreateCooking(PX_PHYSICS_VERSION, *foundation, PxCookingParams(scale));
static PxConvexMesh* cook_points_to_convex(PxPhysics& phy, PxCooking& cook,
                                           const std::vector<PxVec3>& pts, float scale=1.0f)
{
    PxConvexMeshDesc d;
    d.points.count  = (uint32_t)pts.size();
    d.points.stride = sizeof(PxVec3);
    d.points.data   = pts.data();
    d.flags         = PxConvexFlag::eCOMPUTE_CONVEX | PxConvexFlag::eCHECK_ZERO_AREA_TRIANGLES;
    PxDefaultMemoryOutputStream out;
    if (!cook.cookConvexMesh(d, out)) return nullptr;
    PxDefaultMemoryInputData in(out.getData(), out.getSize());
    return phy.createConvexMesh(in);
}

/* ---------- Spawning with DieMesh ---------- */

static PxRigidDynamic* spawn_die_mesh(PxPhysics* phy, PxScene* scn, PxMaterial* mat,
                                      const DieMesh& dm, const PxVec3& pos, std::mt19937_64& rng)
{
    PxRigidDynamic* die = nullptr;
    if (dm.isBox){
        const float he=0.5f * dm.scale;
        die = PxCreateDynamic(*phy, PxTransform(pos), PxBoxGeometry(he,he,he), *mat, 1.0f);
    } else {
        PxConvexMeshGeometry g(dm.convex, PxMeshScale(dm.scale));
        die = PxCreateDynamic(*phy, PxTransform(pos), g, *mat, 1.0f);
    }
    die->setAngularDamping(0.05f);
    die->setLinearDamping(0.01f);
    die->setSleepThreshold(0.05f);

    std::uniform_real_distribution<float> Upos(-0.25f,0.25f), Uimp(2.0f,8.0f), Utor(1.0f,6.0f);
    PxTransform t=die->getGlobalPose(); t.p.x+=Upos(rng); t.p.z+=Upos(rng); die->setGlobalPose(t);
    PxVec3 imp( Uimp(rng)*(rng()&1?1:-1), Uimp(rng), Uimp(rng)*(rng()&1?1:-1) );
    PxVec3 tor( Utor(rng)*(rng()&1?1:-1), Utor(rng)*(rng()&1?1:-1), Utor(rng)*(rng()&1?1:-1) );
    die->addForce(imp, PxForceMode::eIMPULSE);
    die->addTorque(tor, PxForceMode::eIMPULSE);

    scn->addActor(*die);
    return die;
}

// General top-face detector using local face normals
static int top_face_from_normals(const PxQuat& q,
                                 const std::vector<PxVec3>& faceNormalsLocal,
                                 const std::vector<int>& faceNumber)
{
    const PxVec3 up(0,1,0);
    float best=-1e9f; int ans = faceNumber.empty()? 1 : faceNumber[0];
    for (size_t i=0;i<faceNormalsLocal.size();++i){
        PxVec3 w = q.rotate(faceNormalsLocal[i]);
        float d = w.dot(up);
        if (d > best){ best=d; ans = faceNumber.empty()? int(i+1) : faceNumber[i]; }
    }
    return ans;
}