#include <PxPhysicsAPI.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>

using namespace physx;

// Global PhysX variables
PxDefaultAllocator gAllocator;
PxDefaultErrorCallback gErrorCallback;
PxFoundation* gFoundation = nullptr;
PxPhysics* gPhysics = nullptr;
PxScene* gScene = nullptr;
PxMaterial* gMaterial = nullptr;

// Randomness results map
std::unordered_map<int, int> rollResults;

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

// Create a die
PxRigidDynamic* createDie(int sides, PxVec3 position) {
    PxRigidDynamic* die = PxCreateDynamic(*gPhysics, PxTransform(position),
                                          PxBoxGeometry(0.5f, 0.5f, 0.5f), *gMaterial, 10.0f);

    die->setName(std::to_string(sides).c_str()); // Name the die by its number of sides
    gScene->addActor(*die);

    // Apply random force and torque
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> forceDist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> torqueDist(-5.0f, 5.0f);

    die->addForce(PxVec3(forceDist(gen), forceDist(gen), forceDist(gen)), PxForceMode::eIMPULSE);
    die->addTorque(PxVec3(torqueDist(gen), torqueDist(gen), torqueDist(gen)), PxForceMode::eIMPULSE);

    return die;
}

// Simulate the physics scene
void simulateScene(int steps = 500) {
    for (int i = 0; i < steps; ++i) {
        gScene->simulate(1.0f / 60.0f);
        gScene->fetchResults(true);
    }
}

// Detect the top face of the die
int detectTopFace(PxRigidDynamic* die, int sides) {
    PxTransform transform = die->getGlobalPose();
    PxVec3 position = transform.p;

    PxRaycastBuffer hit;
    if (gScene->raycast(PxVec3(position.x, position.y - 1, position.z), PxVec3(0.0f, 1.0f, 0.0f), 1.0f, hit)) {
        PxRigidActor* hitActor = hit.block.actor;
        if (hitActor) {
            // The actor's name represents the face number
            return std::stoi(hitActor->getName());
        }
    }

    // If no face detected, return a random face as fallback
    return std::rand() % sides + 1;
}

// Perform multiple rolls and collect statistics
void rollDice(int sides, int rolls) {
    for (int i = 0; i < rolls; ++i) {
        // Create a die and simulate
        PxRigidDynamic* die = createDie(sides, PxVec3(0, 5, 0));
        simulateScene();

        // Detect the top face
        int result = detectTopFace(die, sides);
        rollResults[result]++;

        // Clean up the die actor
        gScene->removeActor(*die);
        die->release();
    }
}

// Print the analysis of randomness
void analyzeRandomness(int sides, int rolls) {
    std::cout << "Randomness Analysis for " << rolls << " rolls of a " << sides << "-sided die:\n";
    for (int i = 1; i <= sides; ++i) {
        double frequency = (double)rollResults[i] / rolls * 100.0;
        std::cout << "Face " << i << ": " << rollResults[i] << " rolls (" << frequency << "%)\n";
    }
}

// Clean up PhysX
void cleanupPhysX() {
    gScene->release();
    gPhysics->release();
    gFoundation->release();
}

// Main function
int main() {
    srand(static_cast<unsigned>(time(0))); // Seed RNG

    const int sides = 6; // Number of sides on the die
    const int rolls = 1000; // Number of rolls

    initPhysX();

    // Perform dice rolls and analyze randomness
    rollDice(sides, rolls);
    analyzeRandomness(sides, rolls);

    cleanupPhysX();
    return 0;
}