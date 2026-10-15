cmake_minimum_required(VERSION 3.20)
project(code_noodling LANGUAGES C CXX)

# =========================
# Options
# =========================
option(BUILD_CUDA_SIEVE_MGPU "Build the multi-GPU CUDA sieve (K80-optimized)" ON)
option(BUILD_PHYSX_DICE      "Build PhysX dice simulator (D6/D8/D12/D20 + chute + PVD)" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =========================
# CPU targets
# =========================

# OSE (CPU) — your original prime code if you want to keep it building
if (EXISTS "${CMAKE_SOURCE_DIR}/OSE.c")
  add_executable(ose_cpu OSE.c)
  target_compile_options(ose_cpu PRIVATE -O3 -march=native)
endif()

# Unbiased dice engine (any n, NdM+K, JSON/CSV, chi-square)
if (EXISTS "${CMAKE_SOURCE_DIR}/dice_cpu.cpp")
  add_executable(dice_cpu dice_cpu.cpp)
  target_compile_options(dice_cpu PRIVATE -O3 -march=native)
elseif (EXISTS "${CMAKE_SOURCE_DIR}/dice roll improved.cpp")
  add_executable(dice_cpu "dice roll improved.cpp")
  target_compile_options(dice_cpu PRIVATE -O3 -march=native)
endif()

# =========================
# CUDA multi-GPU sieve
# =========================
if (BUILD_CUDA_SIEVE_MGPU)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # Default to Tesla K80 arch (sm_37); you can override at configure time:
    #   cmake .. -DCMAKE_CUDA_ARCHITECTURES=37
    if (NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 37)
    endif()

    if (EXISTS "${CMAKE_SOURCE_DIR}/cuda_sieve_mgpu.cu")
      add_executable(cuda_sieve_mgpu cuda_sieve_mgpu.cu)
      # Threads-per-block default can be overridden: -DTPB=512
      target_compile_definitions(cuda_sieve_mgpu PRIVATE TPB=256)
      target_compile_options(cuda_sieve_mgpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
      )
    endif()

    # If you keep a legacy OSE_CUDA.cc, force CUDA language for it:
    if (EXISTS "${CMAKE_SOURCE_DIR}/OSE_CUDA.cc")
      set_source_files_properties(OSE_CUDA.cc PROPERTIES LANGUAGE CUDA)
      add_executable(ose_cuda OSE_CUDA.cc)
      target_compile_options(ose_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
    endif()
  else()
    message(WARNING "CUDA compiler not found; skipping CUDA targets.")
  endif()
endif()

# =========================
# PhysX dice simulator
# =========================
if (BUILD_PHYSX_DICE AND EXISTS "${CMAKE_SOURCE_DIR}/physx_dice_multi.cpp")
  add_executable(physx_dice_multi physx_dice_multi.cpp)
  target_compile_options(physx_dice_multi PRIVATE -O3 -march=native)

  # ---- PhysX SDK discovery (adjust to your install) ----
  # Optionally set PHYSX_ROOT yourself:
  #   cmake .. -DPHYSX_ROOT=/opt/NVIDIA/PhysX
  if (NOT DEFINED PHYSX_ROOT)
    set(PHYSX_ROOT "" CACHE PATH "Root directory of PhysX SDK (optional)")
  endif()

  find_path(PHYSX_INCLUDE_DIR PxPhysicsAPI.h
    HINTS "${PHYSX_ROOT}/include" ENV CPATH
          /usr/include /usr/local/include /opt/physx/include)

  find_library(PHYSX_LIB            NAMES PhysX_64 PhysX)
  find_library(PHYSX_FOUNDATION_LIB NAMES PhysXFoundation_64 PhysXFoundation)
  find_library(PHYSX_COMMON_LIB     NAMES PhysXCommon_64 PhysXCommon)
  find_library(PHYSX_EXT_LIB        NAMES PhysXExtensions_static PhysXExtensions)
  find_library(PHYSX_COOKING_LIB    NAMES PhysXCooking_64 PhysXCooking)
  find_library(PHYSX_PVD_LIB        NAMES PhysXPvdSDK_static PhysXPvdSDK)

  if (PHYSX_INCLUDE_DIR AND PHYSX_LIB AND PHYSX_FOUNDATION_LIB AND PHYSX_COMMON_LIB AND PHYSX_EXT_LIB AND PHYSX_COOKING_LIB AND PHYSX_PVD_LIB)
    target_include_directories(physx_dice_multi PRIVATE ${PHYSX_INCLUDE_DIR})
    target_link_libraries(physx_dice_multi PRIVATE
      ${PHYSX_LIB}
      ${PHYSX_FOUNDATION_LIB}
      ${PHYSX_COMMON_LIB}
      ${PHYSX_EXT_LIB}
      ${PHYSX_COOKING_LIB}
      ${PHYSX_PVD_LIB}
    )
  else()
    message(WARNING "PhysX not fully found; will try to build but linking may fail.
    Found:
      PHYSX_INCLUDE_DIR=${PHYSX_INCLUDE_DIR}
      PHYSX_LIB=${PHYSX_LIB}
      PHYSX_FOUNDATION_LIB=${PHYSX_FOUNDATION_LIB}
      PHYSX_COMMON_LIB=${PHYSX_COMMON_LIB}
      PHYSX_EXT_LIB=${PHYSX_EXT_LIB}
      PHYSX_COOKING_LIB=${PHYSX_COOKING_LIB}
      PHYSX_PVD_LIB=${PHYSX_PVD_LIB}")
  endif()
endif()

# =========================
# Summary
# =========================
message(STATUS "------------------------------------------------------------")
message(STATUS "Targets:")
get_property(ALL_TGTS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
foreach(t ${ALL_TGTS})
  message(STATUS "  - ${t}")
endforeach()
message(STATUS "Options: BUILD_CUDA_SIEVE_MGPU=${BUILD_CUDA_SIEVE_MGPU}  BUILD_PHYSX_DICE=${BUILD_PHYSX_DICE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "------------------------------------------------------------")