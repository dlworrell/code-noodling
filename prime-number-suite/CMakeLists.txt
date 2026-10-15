cmake_minimum_required(VERSION 3.20)
project(prime_number_suite LANGUAGES C CXX)

option(BUILD_CUDA_SIEVE_MGPU "Build the multi-GPU CUDA sieve (K80-optimized)" ON)
option(BUILD_PHYSX_DICE "Build the maintained PhysX dice simulator" ON)
# Host-specific tuning is opt-in so CI and shared binaries stay portable.
option(NATIVE_ARCH "Compile host code with -march=native" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Apply a consistent reviewer-facing warning baseline to host code and keep the
# historical -O3 for every build type except Debug, so sanitizer and debugger
# builds retain their expected behavior.
function(configure_host_target target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic
      $<$<NOT:$<CONFIG:Debug>>:-O3>
    )
    if(NATIVE_ARCH)
      target_compile_options(${target} PRIVATE -march=native)
    endif()
  endif()
endfunction()

# CPU prime generator.
add_executable(ose_cpu generators/ose.c)
configure_host_target(ose_cpu)
if(NOT MSVC)
  target_link_libraries(ose_cpu PRIVATE m)
endif()

# CPU dice engine and prime-list consumer. The same source is also packaged
# with Autotools in dice/.
add_executable(dice_cpu dice/src/dice_cpu.cpp)
configure_host_target(dice_cpu)

# Optional multi-GPU prime generator. Configuration remains successful on
# machines without CUDA so the portable CPU targets can still be reviewed.
if(BUILD_CUDA_SIEVE_MGPU)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      # Tesla K80 (sm_37) is the primary target and must stay supported; it
      # needs CUDA <= 11.x. Override for newer GPUs, for example
      # -DCMAKE_CUDA_ARCHITECTURES="75;86".
      set(CMAKE_CUDA_ARCHITECTURES 37)
    endif()
    enable_language(CUDA)

    add_executable(cuda_sieve_mgpu generators/cuda_sieve_mgpu.cu)
    set_target_properties(cuda_sieve_mgpu PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
    )
    # The K80 implementation is tuned around 256 threads per block.
    target_compile_definitions(cuda_sieve_mgpu PRIVATE TPB=256)
    target_compile_options(cuda_sieve_mgpu PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
  else()
    message(WARNING "CUDA compiler not found; skipping cuda_sieve_mgpu.")
  endif()
endif()

# Optional maintained PhysX simulator. The prototypes in legacy/ are kept for
# review and are intentionally not build targets.
if(BUILD_PHYSX_DICE)
  set(PHYSX_ROOT "" CACHE PATH "Root directory of the NVIDIA PhysX SDK")
  find_path(PHYSX_INCLUDE_DIR PxPhysicsAPI.h
    HINTS "${PHYSX_ROOT}/include" ENV CPATH
    PATHS /usr/include /usr/local/include /opt/physx/include
  )
  find_library(PHYSX_LIB
    NAMES PhysX_64 PhysX
    HINTS "${PHYSX_ROOT}/lib" "${PHYSX_ROOT}/bin"
  )
  find_library(PHYSX_FOUNDATION_LIB
    NAMES PhysXFoundation_64 PhysXFoundation
    HINTS "${PHYSX_ROOT}/lib" "${PHYSX_ROOT}/bin"
  )
  find_library(PHYSX_COMMON_LIB
    NAMES PhysXCommon_64 PhysXCommon
    HINTS "${PHYSX_ROOT}/lib" "${PHYSX_ROOT}/bin"
  )
  find_library(PHYSX_EXT_LIB
    NAMES PhysXExtensions_static PhysXExtensions
    HINTS "${PHYSX_ROOT}/lib" "${PHYSX_ROOT}/bin"
  )
  find_library(PHYSX_COOKING_LIB
    NAMES PhysXCooking_64 PhysXCooking
    HINTS "${PHYSX_ROOT}/lib" "${PHYSX_ROOT}/bin"
  )
  find_library(PHYSX_PVD_LIB
    NAMES PhysXPvdSDK_static PhysXPvdSDK
    HINTS "${PHYSX_ROOT}/lib" "${PHYSX_ROOT}/bin"
  )

  if(PHYSX_INCLUDE_DIR AND PHYSX_LIB AND PHYSX_FOUNDATION_LIB AND
     PHYSX_COMMON_LIB AND PHYSX_EXT_LIB AND PHYSX_COOKING_LIB AND
     PHYSX_PVD_LIB)
    add_executable(physx_dice_multi physx/physx_dice_multi.cpp)
    configure_host_target(physx_dice_multi)
    target_include_directories(physx_dice_multi PRIVATE
      "${PHYSX_INCLUDE_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/physx"
    )
    target_link_libraries(physx_dice_multi PRIVATE
      "${PHYSX_LIB}"
      "${PHYSX_FOUNDATION_LIB}"
      "${PHYSX_COMMON_LIB}"
      "${PHYSX_EXT_LIB}"
      "${PHYSX_COOKING_LIB}"
      "${PHYSX_PVD_LIB}"
    )
  else()
    message(WARNING "PhysX SDK not fully found; skipping physx_dice_multi.")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME prime-number-suite.ose
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ose.py"
              "$<TARGET_FILE:ose_cpu>"
    )
    add_test(
      NAME prime-number-suite.dice-cpu
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dice_cpu.py"
              "$<TARGET_FILE:dice_cpu>"
              "$<TARGET_FILE:ose_cpu>"
    )
  else()
    message(WARNING "Python 3 not found; skipping integration tests.")
  endif()
endif()

message(STATUS "BUILD_CUDA_SIEVE_MGPU=${BUILD_CUDA_SIEVE_MGPU}")
message(STATUS "BUILD_PHYSX_DICE=${BUILD_PHYSX_DICE}")
message(STATUS "NATIVE_ARCH=${NATIVE_ARCH}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
