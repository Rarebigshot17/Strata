# Strata: GLM-5.3-Flash on NVIDIA GPUs and system RAM.
#
#   strata-glm      the engine (CUDA): every layer on the GPUs, the routed experts in RAM with the most used
#                   cached in VRAM, the rest computed on the CPU. serve/server.py drives it.
#   strata-glm-ref  the same model on the CPU, one token at a time: the reference the GPU engine is checked
#                   against (tools/glm53_*.sh), and an engine for tests without a GPU.
#
#   cmake -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j
#
# ggml (CPU only, static) comes from a llama.cpp checkout (-DSTRATA_GGML_DIR=...) or is fetched at the pinned
# commit: the CPU expert dot products and the CPU reference use it.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.2.0 LANGUAGES C CXX)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")

option(STRATA_PORTABLE "ggml-cpu for any AVX2 CPU instead of this machine's" OFF)
option(STRATA_ENABLE_CUDA "build strata-glm (CUDA); strata-glm-ref (CPU) is always built" OFF)
option(STRATA_BUILD_TESTS "build the engine's unit tests (tests/glm, run with ctest)" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
find_package(Threads REQUIRED)

# ---------------------------------------------------------------- ggml (CPU only, static)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()
if(STRATA_GGML_DIR)
  add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
else()
  include(FetchContent)
  FetchContent_Declare(strata_llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d      # third_party/ggml/VERSION.txt
    GIT_SHALLOW FALSE)
  FetchContent_GetProperties(strata_llamacpp)
  if(NOT strata_llamacpp_POPULATED)
    FetchContent_Populate(strata_llamacpp)
  endif()
  add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
endif()

# ---------------------------------------------------------------- the model, the CPU reference, the serve loop
add_library(strata_glm STATIC
  src/glm/model.cpp src/glm/cpu_ref.cpp src/glm/cpu_experts.cpp src/glm/sampler.cpp src/glm/serve_loop.cpp)
target_include_directories(strata_glm PUBLIC include)
target_link_libraries(strata_glm PUBLIC ggml-cpu ggml-base Threads::Threads)

add_executable(strata-glm-ref src/glm/ref_main.cpp)
target_link_libraries(strata-glm-ref PRIVATE strata_glm)

# ---------------------------------------------------------------- the CUDA engine
if(STRATA_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures (native: the GPUs in this PC)" FORCE)
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")       # 86-real / 86-virtual are still 86
    if(_base MATCHES "^[0-9]+$" AND _base LESS 75)
      message(FATAL_ERROR "strata-glm needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 / 30 / 40 / "
                          "50 series); CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'")
    endif()
  endforeach()
  message(STATUS "strata-glm: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")

  # the quantized GEMV kernels (from llama.cpp's MMVQ) and their dequantizers
  add_library(strata_glm_kernels STATIC
    src/kernels/cuda/native_mmvq.cu src/kernels/cuda/iq_kernels.cu src/kernels/cuda/pdl.cu)
  target_include_directories(strata_glm_kernels PUBLIC include PRIVATE third_party/ggml)
  set_source_files_properties(src/kernels/cuda/native_mmvq.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math")
  target_link_libraries(strata_glm_kernels PUBLIC CUDA::cudart)

  add_library(strata_glm_gpu STATIC
    src/glm/gpu_kernels.cu src/glm/gpu_linear.cu src/glm/gpu_gemm.cu src/glm/gpu_prefill.cu src/glm/expert_cache.cu
    src/glm/engine.cu src/glm/engine_prefill.cu)
  target_link_libraries(strata_glm_gpu PUBLIC strata_glm strata_glm_kernels CUDA::cudart CUDA::cublas)
  target_compile_options(strata_glm_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

  add_executable(strata-glm src/glm/glm_main.cpp)
  target_link_libraries(strata-glm PRIVATE strata_glm_gpu)
endif()

# ---------------------------------------------------------------- tests (no GPU, no model file)
if(STRATA_BUILD_TESTS)
  enable_testing()
  add_executable(glm_sampler_test tests/glm/sampler_test.cpp)
  target_link_libraries(glm_sampler_test PRIVATE strata_glm)
  add_test(NAME glm_sampler COMMAND glm_sampler_test)
  add_executable(glm_serve_loop_test tests/glm/serve_loop_test.cpp)
  target_link_libraries(glm_serve_loop_test PRIVATE strata_glm)
  add_test(NAME glm_serve_loop COMMAND glm_serve_loop_test)
endif()
