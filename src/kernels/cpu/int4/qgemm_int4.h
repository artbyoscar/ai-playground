#pragma once
#include <cstdint>

/*
  Build-time tunables (defaults if not provided by the build system):
  - INT4_ASSERTS        : enable extra runtime checks (1=on, 0=off)
  - INT4_BENCH_JSON     : enable JSON fields in perf harness (1=on, 0=off)
  - INT4_PREFETCH_AHEAD : prefetch distance for A (bytes), e.g. 64/128
*/
#ifndef INT4_ASSERTS
#define INT4_ASSERTS 1
#endif
#ifndef INT4_BENCH_JSON
#define INT4_BENCH_JSON 0
#endif
#ifndef INT4_PREFETCH_AHEAD
#define INT4_PREFETCH_AHEAD 128
#endif

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Kernels (weights: INT4 with per-group fp16 scales; activations: fp16)
// Layout: columns packed in K along groups of `group_size` (e.g., 64).
// Scales are fp16 (uint16_t bits), one per group.
// -----------------------------------------------------------------------------

// Baseline kernel (single-thread)
void qgemm_int4_fp16(
  const uint16_t* A_fp16,          int lda,
  const uint8_t*  B_packed,        const uint16_t* B_scales, int ldb_packed,
  uint16_t*       C_fp16,          int ldc,
  int M, int N, int K, int group_size);

// Multithreaded wrapper (splits work across columns)
void qgemm_int4_fp16_mt(
  const uint16_t* A_fp16,          int lda,
  const uint8_t*  B_packed,        const uint16_t* B_scales, int ldb_packed,
  uint16_t*       C_fp16,          int ldc,
  int M, int N, int K, int group_size,
  int num_threads);

// Tiled MT (reuses A across N-tiles)
void qgemm_int4_fp16_tiled_mt(
  const uint16_t* A_fp16,          int lda,
  const uint8_t*  B_packed,        const uint16_t* B_scales, int ldb_packed,
  uint16_t*       C_fp16,          int ldc,
  int M, int N, int K, int group_size,
  int num_threads, int nc_tile);

// Tiled MT with M-blocking (experimental; adds m_tile)
void qgemm_int4_fp16_tiled_mt_mblocked(
  const uint16_t* A_fp16,          int lda,
  const uint8_t*  B_packed,        const uint16_t* B_scales, int ldb_packed,
  uint16_t*       C_fp16,          int ldc,
  int M, int N, int K, int group_size,
  int num_threads, int nc_tile, int m_tile);

// Optional Q8 path (if you experiment with 8-bit packed weights)
void qgemm_int4_fp16_q8_mt(
  const uint16_t* A_fp16,          int lda,
  const int8_t*   B_packed_q8,     const uint16_t* B_scales, int ldb_q8,
  uint16_t*       C_fp16,          int ldc,
  int M, int N, int K, int group_size,
  int num_threads);

// Fused epilogue variant (with bias and activation)
#ifdef INT4_FUSE_BIAS
void qgemm_int4_fp16_tiled_mt_fused(
    const uint16_t* A, int lda,
    const uint8_t* B_packed, 
    const uint16_t* scales, int ldb,
    uint16_t* C, int ldc,
    int M, int N, int K, int group_size,
    int nc_tile, int num_threads,
    const uint16_t* bias_fp16,
    int activation);
#endif

#ifdef __cplusplus
} // extern "C"
#endif