cmake_minimum_required(VERSION 3.22)
project(qgemm_int4_cpu LANGUAGES C CXX)

# ---- Toolchain / Defaults ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Options ----
option(BUILD_DML "Build DirectML smoke targets" OFF)
option(INT4_FUSE_BIAS "Enable fused bias/activation epilogue in kernels" OFF)

# ---- Global CPU Flags ----
if (MSVC)
  add_compile_options(/O2 /fp:fast /arch:AVX2)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-O3 -mavx2 -mfma -mf16c -ffp-contract=fast)
endif()

# Apply INT4_FUSE_BIAS definition if enabled
if (INT4_FUSE_BIAS)
  add_compile_definitions(INT4_FUSE_BIAS=1)
endif()

# ---- Main Library ----
add_library(qgemm_int4 STATIC
  qgemm_int4.cpp
  pack_int4.cpp
  pack_q8.cpp
)
target_include_directories(qgemm_int4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- Pack Loader Library ----
add_library(pack_loader STATIC
  pack_loader.cpp
)
target_include_directories(pack_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- Core Performance Tests ----
add_executable(test_qgemm tests/test_qgemm.cpp)
target_link_libraries(test_qgemm PRIVATE qgemm_int4)

add_executable(test_qgemm_perf tests/test_perf.cpp)
target_link_libraries(test_qgemm_perf PRIVATE qgemm_int4)

add_executable(test_qgemm_perf_mt tests/test_perf_mt.cpp)
target_link_libraries(test_qgemm_perf_mt PRIVATE qgemm_int4)

add_executable(test_qgemm_perf_tiled_mt tests/test_perf_tiled_mt.cpp)
target_link_libraries(test_qgemm_perf_tiled_mt PRIVATE qgemm_int4)

add_executable(test_qgemm_perf_vs_baseline tests/test_perf_vs_baseline.cpp)
target_link_libraries(test_qgemm_perf_vs_baseline PRIVATE qgemm_int4)

# ---- Q8 Tests ----
add_executable(test_qgemm_perf_q8_mt tests/test_perf_q8_mt.cpp)
target_link_libraries(test_qgemm_perf_q8_mt PRIVATE qgemm_int4)
if (NOT MSVC)
  target_compile_options(test_qgemm_perf_q8_mt PRIVATE -mavx2 -mfma -mf16c -O3)
endif()

add_executable(test_q8_pack tests/test_q8_pack.cpp)
target_link_libraries(test_q8_pack PRIVATE qgemm_int4)

# ---- Packed File Tests ----
add_executable(test_perf_load_packed tests/test_perf_load_packed.cpp)
target_link_libraries(test_perf_load_packed PRIVATE qgemm_int4 pack_loader)

# ---- Fusion/Epilogue Tests ----
add_executable(test_perf_bias tests/test_perf_bias.cpp)
target_link_libraries(test_perf_bias PRIVATE qgemm_int4)
if (NOT MSVC)
  target_compile_options(test_perf_bias PRIVATE -mavx2 -mfma -mf16c -O3)
endif()

# ---- Correctness Tests ----
enable_testing()

add_executable(test_qgemm_correctness tests/test_correctness.cpp)
target_link_libraries(test_qgemm_correctness PRIVATE qgemm_int4)
if (NOT MSVC)
  target_compile_options(test_qgemm_correctness PRIVATE -mavx2 -mfma -mf16c -O3)
endif()

add_test(NAME correctness
  COMMAND test_qgemm_correctness --threshold 7.2e-2
)

# ---- Optional: DML Subtree ----
if (BUILD_DML)
  add_subdirectory(dml/int4)
endif()

# ---- Status Messages ----
message(STATUS "Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  INT4_FUSE_BIAS: ${INT4_FUSE_BIAS}")
message(STATUS "  BUILD_DML: ${BUILD_DML}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")