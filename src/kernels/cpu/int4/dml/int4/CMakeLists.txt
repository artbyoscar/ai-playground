cmake_minimum_required(VERSION 3.22)
project(qgemm_int4_cpu LANGUAGES C CXX)

# ---- toolchain / defaults ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- tunables (can be overridden via -D on configure) ----
option(INT4_ASSERTS       "Enable runtime asserts in kernels" ON)
option(INT4_BENCH_JSON    "Enable JSON output in perf harness" OFF)
set(INT4_PREFETCH_AHEAD "128" CACHE STRING "Prefetch distance for A (bytes)")
option(BUILD_DML          "Build DirectML smoke targets" OFF)
option(INT4_FUSE_BIAS     "Enable fused bias/activation epilogue in kernels" OFF)

message(STATUS
  "INT4_ASSERTS=${INT4_ASSERTS}  "
  "INT4_BENCH_JSON=${INT4_BENCH_JSON}  "
  "INT4_PREFETCH_AHEAD=${INT4_PREFETCH_AHEAD}  "
  "BUILD_DML=${BUILD_DML}  "
  "INT4_FUSE_BIAS=${INT4_FUSE_BIAS}"
)

# ---- global CPU flags (clang/gcc on *nix, clang-cl on Windows) ----
if (MSVC)
  add_compile_options(/O2 /fp:fast /arch:AVX2)
else()
  add_compile_options(-O3 -mavx2 -mfma -mf16c -ffp-contract=fast)
endif()

# ---- directory-wide tunables ----
add_compile_definitions(
  $<$<BOOL:${INT4_ASSERTS}>:INT4_ASSERTS=1>
  $<$<BOOL:${INT4_BENCH_JSON}>:INT4_BENCH_JSON=1>
  INT4_PREFETCH_AHEAD=${INT4_PREFETCH_AHEAD}
  $<$<BOOL:${INT4_FUSE_BIAS}>:INT4_FUSE_BIAS=1>
)

# ---- library ----
add_library(qgemm_int4 STATIC
  qgemm_int4.cpp
  pack_int4.cpp
  qgemm_q8_stub.cpp
)
target_include_directories(qgemm_int4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- tests / perf (CPU) ----
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

# ---- packed file loader + perf that consumes pack/q4edge ----
if (NOT TARGET pack_loader)
  add_library(pack_loader pack_loader.cpp)
  target_include_directories(pack_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  if (MSVC)
    target_compile_definitions(pack_loader PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
endif()

if (NOT TARGET test_perf_load_packed)
  add_executable(test_perf_load_packed tests/test_perf_load_packed.cpp)
  target_link_libraries(test_perf_load_packed PRIVATE qgemm_int4 pack_loader)
endif()

# ---- correctness test (threshold 7.2e-2) ----
enable_testing()

add_executable(test_qgemm_correctness tests/test_correctness.cpp)
target_link_libraries(test_qgemm_correctness PRIVATE qgemm_int4)
if (NOT MSVC)
  target_compile_options(test_qgemm_correctness PRIVATE -mavx2 -mfma -mf16c -O3)
endif()

add_test(NAME correctness
  COMMAND test_qgemm_correctness --threshold 7.2e-2
)

# ---- optional: Q8 perf + bias epilogue timing ----
add_executable(test_qgemm_perf_q8_mt tests/test_perf_q8_mt.cpp)
target_link_libraries(test_qgemm_perf_q8_mt PRIVATE qgemm_int4)
if (NOT MSVC)
  target_compile_options(test_qgemm_perf_q8_mt PRIVATE -mavx2 -mfma -mf16c -O3)
endif()

add_executable(test_perf_bias tests/test_perf_bias.cpp)
target_link_libraries(test_perf_bias PRIVATE qgemm_int4)
if (NOT MSVC)
  target_compile_options(test_perf_bias PRIVATE -mavx2 -mfma -mf16c -O3)
endif()

# ---- DML subtree (off by default) ----
if (BUILD_DML)
  add_subdirectory(dml/int4)
endif()

