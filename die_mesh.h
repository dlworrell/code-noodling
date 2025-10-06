// die_mesh.h — Platonic solids + face normals + convex mesh cooking for PhysX
// Provides: make_d6_box, make_d8, make_d12, make_d20, cook_points_to_convex,
// spawn_die_mesh, top_face_from_normals.
// Works stand-alone; put in the same directory as physx_dice_multi.cpp.

#pragma once
#include <PxPhysicsAPI.h>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <iostream>

using namespace physx;

struct DieMesh {
    PxConvexMesh* convex = nullptr;
    PxGeometryType::Enum geomType = PxGeometryType::eCONVEXMESH;
    float scale = 1.0f;
    bool isBox = false;
    int faces = 0;
    std::vector<PxVec3> faceNormalsLocal;
    std::vector<int> faceNumber;
};

// Normalize a vector
static inline PxVec3 unit(const PxVec3& v){ float m = v.magnitude(); return (m>0)? v*(1.0f/m) : v; }

/* ------------------- DIE GEOMETRIES ------------------- */
static DieMesh make_d6_box(){
    DieMesh dm; dm.isBox = true; dm.geomType = PxGeometryType::eBOX; dm.faces = 6;
    dm.faceNormalsLocal = {
        PxVec3(0,1,0), PxVec3(0,-1,0), PxVec3(1,0,0),
        PxVec3(-1,0,0), PxVec3(0,0,1), PxVec3(0,0,-1)
    };
    dm.faceNumber = {1,6,2,5,3,4};
    return dm;
}

static DieMesh make_d8(PxPhysics& phy, float scale=1.0f){
    DieMesh dm; dm.scale=scale; dm.faces=8;
    dm.faceNormalsLocal = {
        unit(PxVec3(+1,+1, 0)), unit(PxVec3(+1,-1, 0)),
        unit(PxVec3(-1,+1, 0)), unit(PxVec3(-1,-1, 0)),
        unit(PxVec3( 0,+1,+1)), unit(PxVec3( 0,+1,-1)),
        unit(PxVec3( 0,-1,+1)), unit(PxVec3( 0,-1,-1))
    };
    dm.faceNumber = {1,2,3,4,5,6,7,8};
    return dm;
}

static DieMesh make_d12(PxPhysics& phy, float scale=1.0f){
    DieMesh dm; dm.scale=scale; dm.faces=12;
    const float phi = (1.0f + std::sqrt(5.0f))*0.5f;
    dm.faceNormalsLocal = {
        unit(PxVec3( 0, 1, phi)), unit(PxVec3( 0, 1,-phi)),
        unit(PxVec3( 0,-1, phi)), unit(PxVec3( 0,-1,-phi)),
        unit(PxVec3( 1, phi, 0)), unit(PxVec3(-1, phi, 0)),
        unit(PxVec3( 1,-phi, 0)), unit(PxVec3(-1,-phi, 0)),
        unit(PxVec3( phi, 0, 1)), unit(PxVec3( phi, 0,-1)),
        unit(PxVec3(-phi, 0, 1)), unit(PxVec3(-phi, 0,-1))
    };
    dm.faceNumber.resize(12); for(int i=0;i<12;++i) dm.faceNumber[i]=i+1;
    return dm;
}

static DieMesh make_d20(PxPhysics& phy, float scale=1.0f){
    DieMesh dm; dm.scale=scale; dm.faces=20;
    const float phi=(1.0f+std::sqrt(5.0f))*0.5f;
    std::vector<PxVec3> n = {
        { 0,  1,  phi},{ 0,  1, -phi},{ 0, -1,  phi},{ 0, -1, -phi},
        { 1,  phi, 0},{-1,  phi, 0},{ 1, -phi, 0},{-1, -phi, 0},
        { phi, 0,  1},{ phi, 0, -1},{-phi, 0,  1},{-phi, 0, -1},
        { 1, 1, 1},{ 1, 1,-1},{ 1,-1, 1},{ 1,-1,-1},
        {-1, 1, 1},{-1, 1,-1},{-1,-1, 1},{-1,-1,-1}
    };
    dm.faceNormalsLocal.clear();
    for(auto &x:n) dm.faceNormalsLocal.push_back(unit(x));
    dm.faceNumber.resize(20); for(int i=0;i<20;++i) dm.faceNumber[i]=i+1;
    return dm;
}

/* ------------------- CONVEX COOKING ------------------- */
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

/* ------------------- SPAWN & TOP-FACE ------------------- */
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

static int top_face_from_normals(const PxQuat& q,
                                 const std::vector<PxVec3>& faceNormalsLocal,
                                 const std::vector<int>& faceNumber)
{
    const PxVec3 up(0,1,0);
    float best=-1e9f; int ans = 1;
    for (size_t i=0;i<faceNormalsLocal.size();++i){
        PxVec3 w = q.rotate(faceNormalsLocal[i]);
        float d = w.dot(up);
        if (d > best){ best=d; ans = faceNumber.empty()? int(i+1) : faceNumber[i]; }
    }
    return ans;
}