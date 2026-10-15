# Legacy PhysX Dice Prototypes

These files preserve the evolution of the physical-dice experiments but are
not maintained build targets:

| File | Historical purpose | Review limitation |
|---|---|---|
| `dice_roll_with_physX.cpp` | First box-die PhysX proof of concept | No ground plane; actor-name raycast cannot identify a face |
| `dice_roll_improved.cpp` | Statistics added to the box-die prototype | Every die is still a cube; face detection is not validated |
| `physx_dice.cpp` | Prime-seeded D6 experiment with JSON/CSV output | Superseded by the multi-die simulator |

Use [`../physx/physx_dice_multi.cpp`](../physx/physx_dice_multi.cpp) for maintained PhysX
work and [`../dice/src/dice_cpu.cpp`](../dice/src/dice_cpu.cpp) for the portable mathematical dice
engine. Keeping the prototypes outside CMake prevents incomplete experimental
paths from being mistaken for verified executables while leaving them
available for source review and future extraction of useful ideas.
