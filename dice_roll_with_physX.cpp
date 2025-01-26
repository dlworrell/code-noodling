#include <PxPhysicsAPI.h>
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace physx;

// Global PhysX variables
PxDefaultAllocator gAllocator;
PxDefaultErrorCallback gErrorCallback;
PxFoundation* gFoundation = nullptr;
PxPhysics* gPhysics = nullptr;
PxScene* gScene = nullptr;
PxMaterial* gMaterial = nullptr;

// Initialize PhysX
void initPhysX() {
    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
    gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale());

    PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(2);
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    gScene = gPhysics->createScene(sceneDesc);

    gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f); // Default material
}

// Load die model
PxRigidDynamic* createDie(PxVec3 position) {
    PxRigidDynamic* die = PxCreateDynamic(*gPhysics, PxTransform(position),
        PxBoxGeometry(0.5f, 0.5f, 0.5f), *gMaterial, 10.0f);

    die->setName("Die");
    gScene->addActor(*die);

    // Apply random initial force and torque
    die->addForce(PxVec3(rand() % 10 - 5, rand() % 10 + 10, rand() % 10 - 5), PxForceMode::eIMPULSE);
    die->addTorque(PxVec3(rand() % 10 - 5, rand() % 10 - 5, rand() % 10 - 5), PxForceMode::eIMPULSE);

    return die;
}

// Detect the top face
int detectTopFace(PxRigidDynamic* die) {
    PxTransform transform = die->getGlobalPose();
    PxVec3 position = transform.p;

    PxRaycastBuffer hit;
    if (gScene->raycast(PxVec3(position.x, position.y - 1, position.z), PxVec3(0, 1, 0), 1.0f, hit)) {
        return std::stoi(hit.block.actor->getName()); // Assumes face names are numbers
    }

    return -1; // No result
}

// Simulate rolls
void simulateRolls(int numRolls) {
    for (int i = 0; i < numRolls; ++i) {
        PxRigidDynamic* die = createDie(PxVec3(0, 5, 0));

        // Run the simulation
        for (int j = 0; j < 500; ++j) {
            gScene->simulate(1.0f / 60.0f);
            gScene->fetchResults(true);
        }

        // Detect top face
        int result = detectTopFace(die);
        std::cout << "Roll " << i + 1 << ": Top face = " << result << std::endl;
    }
}

// Cleanup PhysX
void cleanupPhysX() {
    gScene->release();
    gPhysics->release();
    gFoundation->release();
}

// Main function
int main() {
    srand(static_cast<unsigned>(time(0))); // Seed RNG
    initPhysX();

    // Simulate 10 rolls
    simulateRolls(10);

    // Cleanup
    cleanupPhysX();
    return 0;
}