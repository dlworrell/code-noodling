cmake_minimum_required(VERSION 3.20)
project(code_noodling LANGUAGES C CXX)

# ---- Options ----
option(BUILD_OSE_CUDA "Build legacy single-GPU OSE CUDA target if available" OFF)
option(BUILD_DICE_PHYSX "Build PhysX-based dice target (requires PhysX SDK/dev libs)" OFF)
option(BUILD_CUDA_SIEVE_MGPU "Build the new multi-GPU CUDA sieve (K80-optimized)" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =========================
# CPU TARGETS
# =========================

# ---- OSE (CPU) ----
add_executable(ose_cpu OSE.c)
target_compile_options(ose_cpu PRIVATE -O3 -march=native)

# ---- Dice (CPU) ----
# Filename contains spaces: "dice roll improved.cpp"
add_executable(dice_cpu "dice roll improved.cpp")
target_compile_options(dice_cpu PRIVATE -O3 -march=native)

# ---- Dice (PhysX) ----
if (BUILD_DICE_PHYSX)
  # You need to point these to your PhysX install; example hints:
  # set(PHYSX_ROOT "/opt/nvidia/PhysX")  # <-- EDIT this for your system
  find_path(PHYSX_INCLUDE_DIR NAMES PxPhysicsAPI.h PATHS
    "${PHYSX_ROOT}/include" "/usr/local/include" "/opt/physx/include" ENV CPATH)
  find_library(PHYSX_LIB NAMES PhysX_64 PhysX PATHS
    "${PHYSX_ROOT}/lib64" "/usr/local/lib" "/opt/physx/lib" ENV LIBRARY_PATH)
  find_library(PHYSX_FOUNDATION_LIB NAMES PhysXFoundation_64 PhysXFoundation PATHS
    "${PHYSX_ROOT}/lib64" "/usr/local/lib" "/opt/physx/lib" ENV LIBRARY_PATH)

  if (PHYSX_INCLUDE_DIR AND PHYSX_LIB AND PHYSX_FOUNDATION_LIB)
    add_executable(dice_physx "dice_roll_with_physX.cpp")
    target_include_directories(dice_physx PRIVATE ${PHYSX_INCLUDE_DIR})
    target_link_libraries(dice_physx PRIVATE ${PHYSX_LIB} ${PHYSX_FOUNDATION_LIB})
    target_compile_options(dice_physx PRIVATE -O3 -march=native)
    target_compile_definitions(dice_physx PRIVATE PHYSX_ENABLED=1)
  else()
    message(WARNING "PhysX not found; disable BUILD_DICE_PHYSX or set PHYSX_ROOT properly.")
  endif()
endif()

# =========================
# CUDA TARGETS
# =========================

if (BUILD_OSE_CUDA OR BUILD_CUDA_SIEVE_MGPU)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # Default CUDA arch for Tesla K80 (Kepler): sm_37
    # Allow override via: -DCMAKE_CUDA_ARCHITECTURES="37" or "37;80" etc.
    if (NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 37)
    endif()
  else()
    message(WARNING "CUDA compiler not found; skipping CUDA targets.")
  endif()
endif()

# ---- Legacy single-GPU OSE CUDA (optional; keep if you still want it) ----
if (BUILD_OSE_CUDA AND CMAKE_CUDA_COMPILER)
  # If you still have OSE_CUDA.cc and want to treat it as CUDA, better to rename to .cu.
  # Otherwise we can force the language as below.
  set(OSE_CUDA_SRC OSE_CUDA.cc)
  set_source_files_properties(${OSE_CUDA_SRC} PROPERTIES LANGUAGE CUDA)
  add_executable(ose_cuda ${OSE_CUDA_SRC})
  set_target_properties(ose_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_compile_options(ose_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
  )
endif()

# ---- New multi-GPU CUDA sieve (K80-optimized) ----
if (BUILD_CUDA_SIEVE_MGPU AND CMAKE_CUDA_COMPILER)
  add_executable(cuda_sieve_mgpu cuda_sieve_mgpu.cu)
  # Let TPB be overridden: -DTPB=512
  target_compile_definitions(cuda_sieve_mgpu PRIVATE TPB=256)
  target_compile_options(cuda_sieve_mgpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endif()

# ---- Convenience: print summary ----
message(STATUS "------------------------------------------------------------")
message(STATUS "Targets:")
get_property(ALL_TGTS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
foreach(t ${ALL_TGTS})
  message(STATUS "  - ${t}")
endforeach()
message(STATUS "Options:")
message(STATUS "  BUILD_OSE_CUDA=${BUILD_OSE_CUDA}")
message(STATUS "  BUILD_CUDA_SIEVE_MGPU=${BUILD_CUDA_SIEVE_MGPU}")
message(STATUS "  BUILD_DICE_PHYSX=${BUILD_DICE_PHYSX}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "------------------------------------------------------------")