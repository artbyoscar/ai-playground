#include "qgemm_int4.h"
#include "parallel_for.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
  #include <intrin.h>
  #include <immintrin.h>
  #include <xmmintrin.h>
#else
  #include <immintrin.h>
  #include <xmmintrin.h>
#endif

// --- Correct fp16 <-> fp32 helpers (F16C fast path, scalar fallback) ---
#if defined(__F16C__)
static inline float fp16_to_fp32(uint16_t h) {
  __m128i hh = _mm_cvtsi32_si128((int)h);
  __m128  f  = _mm_cvtph_ps(hh);
  return _mm_cvtss_f32(f);
}
static inline uint16_t fp16_from_fp32(float f) {
  __m128  x  = _mm_set_ss(f);
  __m128i h  = _mm_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return (uint16_t)_mm_extract_epi16(h, 0);
}
#else
// scalar IEEE-754 half conversion (round-to-nearest-even, clamp to max finite)
static inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign; // +/-0
    } else {
      int e = -1;
      do { mant <<= 1; ++e; } while ((mant & 0x400) == 0);
      mant &= 0x3FF;
      uint32_t exp_f = (127 - 15 - e);
      bits = sign | (exp_f << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13); // inf/NaN
  } else {
    uint32_t exp_f = exp - 15 + 127;
    bits = sign | (exp_f << 23) | (mant << 13);
  }
  float out; std::memcpy(&out, &bits, sizeof(out));
  return out;
}
static inline uint16_t fp16_from_fp32(float f) {
  uint32_t x; std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;

  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    uint32_t t = mant >> (1 - exp + 13);
    if ((mant >> (1 - exp + 12)) & 1u) t += 1u; // RNE
    return (uint16_t)(sign | t);
  } else if (exp >= 31) {
    return (uint16_t)(sign | 0x7BFFu); // clamp to 65504
  } else {
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) half += 1u; // RNE
    return (uint16_t)half;
  }
}
#endif

#if defined(__AVX2__) && defined(__F16C__)
// Convert 64 fp16 → 64 fp32 into a32
static inline void convert_a64_fp16_to_fp32(const uint16_t* a_fp16, float* a32) {
  __m128i h0 = _mm_loadu_si128((const __m128i*)(a_fp16 +  0));
  __m128i h1 = _mm_loadu_si128((const __m128i*)(a_fp16 +  8));
  __m128i h2 = _mm_loadu_si128((const __m128i*)(a_fp16 + 16));
  __m128i h3 = _mm_loadu_si128((const __m128i*)(a_fp16 + 24));
  __m128i h4 = _mm_loadu_si128((const __m128i*)(a_fp16 + 32));
  __m128i h5 = _mm_loadu_si128((const __m128i*)(a_fp16 + 40));
  __m128i h6 = _mm_loadu_si128((const __m128i*)(a_fp16 + 48));
  __m128i h7 = _mm_loadu_si128((const __m128i*)(a_fp16 + 56));
  _mm256_storeu_ps(a32 +  0, _mm256_cvtph_ps(h0));
  _mm256_storeu_ps(a32 +  8, _mm256_cvtph_ps(h1));
  _mm256_storeu_ps(a32 + 16, _mm256_cvtph_ps(h2));
  _mm256_storeu_ps(a32 + 24, _mm256_cvtph_ps(h3));
  _mm256_storeu_ps(a32 + 32, _mm256_cvtph_ps(h4));
  _mm256_storeu_ps(a32 + 40, _mm256_cvtph_ps(h5));
  _mm256_storeu_ps(a32 + 48, _mm256_cvtph_ps(h6));
  _mm256_storeu_ps(a32 + 56, _mm256_cvtph_ps(h7));
}

// Unpack 32 bytes -> 64 signed int8 in qbuf[0..63], mapping 0..15 -> -8..+7
static inline void unpack_q64_avx2(const uint8_t* src32B, int8_t* qbuf64) {
  // Interleaved order: q0 (high nibble) then q1 (low nibble) for each byte
  for (int b=0; b<32; ++b) {
    uint8_t v = src32B[b];
    qbuf64[2*b+0] = (int8_t)(((v >> 4) & 0xF) - 8);
    qbuf64[2*b+1] = (int8_t)(((v     ) & 0xF) - 8);
  }
}

// A[64 half] dot (scale * qint4[64]) → float
static inline float dot64_avx2_f16_q4(const uint16_t* a_fp16, const uint8_t* packed32, float scale) {
  alignas(32) int8_t qbuf[64];
  unpack_q64_avx2(packed32, qbuf);

  const __m256 vscale = _mm256_set1_ps(scale);
  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < 64; i += 8) {
    __m128i ah = _mm_loadu_si128((const __m128i*)(a_fp16 + i));   // 8 * fp16
    __m256  af = _mm256_cvtph_ps(ah);                             // 8 * fp32

    __m128i q8   = _mm_loadl_epi64((const __m128i*)(qbuf + i));   // 8 bytes
    __m256i q32  = _mm256_cvtepi8_epi32(q8);
    __m256  qf   = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), vscale);

    acc = _mm256_fmadd_ps(af, qf, acc);
  }

  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float out; _mm_store_ss(&out, sum);
  return out;
}

// dot using preconverted a32[64]
static inline float dot64_from_a32_avx2(const float* a32, const uint8_t* packed32, float scale) {
  alignas(32) int8_t qbuf[64];
  unpack_q64_avx2(packed32, qbuf);

  const __m256 vscale = _mm256_set1_ps(scale);
  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < 64; i += 8) {
    __m256 af = _mm256_loadu_ps(a32 + i);

    __m128i q8   = _mm_loadl_epi64((const __m128i*)(qbuf + i));
    __m256i q32  = _mm256_cvtepi8_epi32(q8);
    __m256  qf   = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), vscale);

    acc = _mm256_fmadd_ps(af, qf, acc);
  }

  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float out; _mm_store_ss(&out, sum);
  return out;
}
#endif // __AVX2__ && __F16C__

// Scalar fallback for a single 64-chunk (used for tails or no-AVX builds)
static inline float dot64_scalar_f16_q4(const uint16_t* a_fp16, const uint8_t* packed32, float scale) {
  float acc = 0.0f;
  int k = 0;
  for (int b=0; b<32; ++b) {
    uint8_t packed = packed32[b];
    int q0 =  (packed >> 4) & 0xF; q0 -= 8;
    int q1 =  (packed     ) & 0xF; q1 -= 8;
    acc += fp16_to_fp32(a_fp16[k])   * (scale * (float)q0);
    acc += fp16_to_fp32(a_fp16[k+1]) * (scale * (float)q1);
    k += 2;
  }
  return acc;
}

void qgemm_int4_fp16(
  const uint16_t* A_fp16, int lda,
  const uint8_t*  B_packed, const uint16_t* B_scales, int /*ldb_packed*/,
  uint16_t* C_fp16, int ldc,
  int M, int N, int K, int group_size)
{
  const int groups = (K + group_size - 1) / group_size;
  const int bytes_per_group = group_size / 2;

  const bool can_use_avx2 =
  #if defined(__AVX2__) && defined(__F16C__)
    (group_size == 64);
  #else
    false;
  #endif

  for (int m=0; m<M; ++m) {
    const uint16_t* Arow = A_fp16 + m*lda;

    for (int n=0; n<N; ++n) {
      float acc = 0.0f;

      const uint8_t*  Bcol = B_packed + n * (groups * bytes_per_group);
      const uint16_t* Sc   = B_scales + n * groups;

      int k = 0;
      for (int g=0; g<groups; ++g) {
        const float s = fp16_to_fp32(Sc[g]);
        const uint8_t* group_ptr = Bcol + g * bytes_per_group;

        #if defined(__AVX2__)
        _mm_prefetch((const char*)(group_ptr + 64), _MM_HINT_T0);
        #endif

        if (can_use_avx2 && (k + 64) <= K) {
        #if defined(__AVX2__) && defined(__F16C__)
          acc += dot64_avx2_f16_q4(Arow + k, group_ptr, s);
          k += 64;
          continue;
        #endif
        }

        // Fallback scalar for tails / non-AVX2 builds
        for (int b=0; b<bytes_per_group; ++b) {
          uint8_t packed = group_ptr[b];
          int q0 =  (packed >> 4) & 0xF; q0 -= 8;
          int q1 =  (packed     ) & 0xF; q1 -= 8;
          if (k < K)     acc += fp16_to_fp32(Arow[k])   * (s * (float)q0);
          if (k+1 < K)   acc += fp16_to_fp32(Arow[k+1]) * (s * (float)q1);
          k += 2;
        }
      }
      C_fp16[m*ldc + n] = fp16_from_fp32(acc);
    }
  }
}

// Multithreaded wrapper: split work across columns (N)
void qgemm_int4_fp16_mt(
  const uint16_t* A_fp16, int lda,
  const uint8_t*  B_packed, const uint16_t* B_scales, int ldb_packed,
  uint16_t* C_fp16, int ldc,
  int M, int N, int K, int group_size,
  int num_threads)
{
  (void)ldb_packed;

  if (num_threads <= 0) {
    num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads <= 0) num_threads = 4;
  }
  num_threads = std::max(1, std::min(num_threads, N));

  const int groups = (K + group_size - 1) / group_size;
  const int bytes_per_group = group_size / 2;

  const bool can_use_avx2 =
  #if defined(__AVX2__) && defined(__F16C__)
    (group_size == 64);
  #else
    false;
  #endif

  parallel_for(0, N, num_threads, [&](int n0, int n1){
    for (int n=n0; n<n1; ++n) {
      const uint8_t*  Bcol = B_packed + n * (groups * bytes_per_group);
      const uint16_t* Sc   = B_scales + n * groups;

      for (int m=0; m<M; ++m) {
        const uint16_t* Arow = A_fp16 + m*lda;
        float acc = 0.0f;
        int k = 0;

        for (int g=0; g<groups; ++g) {
          const float s = fp16_to_fp32(Sc[g]);
          const uint8_t* group_ptr = Bcol + g * bytes_per_group;

          #if defined(__AVX2__)
          _mm_prefetch((const char*)(group_ptr + 64), _MM_HINT_T0);
          #endif

          if (can_use_avx2 && (k + 64) <= K) {
          #if defined(__AVX2__) && defined(__F16C__)
            acc += dot64_avx2_f16_q4(Arow + k, group_ptr, s);
            k += 64;
            continue;
          #endif
          }

          for (int b=0; b<bytes_per_group; ++b) {
            uint8_t packed = group_ptr[b];
            int q0 =  (packed >> 4) & 0xF; q0 -= 8;
            int q1 =  (packed     ) & 0xF; q1 -= 8;
            if (k < K)     acc += fp16_to_fp32(Arow[k])   * (s * (float)q0);
            if (k+1 < K)   acc += fp16_to_fp32(Arow[k+1]) * (s * (float)q1);
            k += 2;
          }
        }
        C_fp16[m*ldc + n] = fp16_from_fp32(acc);
      }
    }
  });
}

// --------- Tiled MT kernel (reuses A across NC columns) ----------
void qgemm_int4_fp16_tiled_mt(
  const uint16_t* A_fp16, int lda,
  const uint8_t*  B_packed, const uint16_t* B_scales, int ldb_packed,
  uint16_t*       C_fp16, int ldc,
  int M, int N, int K, int group_size,
  int num_threads, int nc_tile)
{
  (void)ldb_packed;

  if (num_threads <= 0) {
    num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads <= 0) num_threads = 4;
  }
  if (nc_tile <= 0) nc_tile = 8;
  const int groups = (K + group_size - 1) / group_size;
  const int bytes_per_group = group_size / 2;

  const bool can_use_avx2 =
  #if defined(__AVX2__) && defined(__F16C__)
    (group_size == 64);
  #else
    false;
  #endif

  const int num_tiles = (N + nc_tile - 1) / nc_tile;
  num_threads = std::max(1, std::min(num_threads, num_tiles));

  parallel_for(0, num_tiles, num_threads, [&](int t0, int t1){
    alignas(32) float a32[64]; // reused per 64-chunk
    for (int t = t0; t < t1; ++t) {
      const int n_begin = t * nc_tile;
      const int n_end   = std::min(N, n_begin + nc_tile);
      const int cols    = n_end - n_begin;

      for (int m=0; m<M; ++m) {
        float acc[32]; // supports nc_tile up to 32
        for (int i=0; i<cols; ++i) acc[i] = 0.0f;

        const uint16_t* Arow = A_fp16 + m*lda;

        int k = 0;
        for (int g=0; g<groups; ++g) {
          const int k_rem = K - k;

          if (can_use_avx2 && k_rem >= 64) {
            #if defined(__AVX2__) && defined(__F16C__)
            convert_a64_fp16_to_fp32(Arow + k, a32);

            for (int i=0; i<cols; ++i) {
              const int n = n_begin + i;
              const uint8_t*  group_ptr = B_packed + n * (groups * bytes_per_group) + g * bytes_per_group;
              const float     s = fp16_to_fp32(B_scales[n * groups + g]);

              acc[i] += dot64_from_a32_avx2(a32, group_ptr, s);
            }
            k += 64;
            continue;
            #endif
          }

          // Scalar tail or no-AVX build
          for (int i=0; i<cols; ++i) {
            const int n = n_begin + i;
            const uint8_t*  group_ptr = B_packed + n * (groups * bytes_per_group) + g * bytes_per_group;
            const float     s = fp16_to_fp32(B_scales[n * groups + g]);

            int kk = k;
            for (int b=0; b<bytes_per_group; ++b) {
              uint8_t packed = group_ptr[b];
              int q0 =  (packed >> 4) & 0xF; q0 -= 8;
              int q1 =  (packed     ) & 0xF; q1 -= 8;
              if (kk < K)     acc[i] += fp16_to_fp32(Arow[kk])   * (s * (float)q0);
              if (kk+1 < K)   acc[i] += fp16_to_fp32(Arow[kk+1]) * (s * (float)q1);
              kk += 2;
            }
          }
          k += 64; // safe even for last partial group
        }

        for (int i=0; i<cols; ++i) {
          const int n = n_begin + i;
          C_fp16[m*ldc + n] = fp16_from_fp32(acc[i]);
        }
      }
    }
  });
}

#if defined(__AVX2__)
static inline float dot64_from_a32_q8_avx2(const float* a32, const int8_t* q8_64, float scale) {
  const __m256 vscale = _mm256_set1_ps(scale);
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < 64; i += 8) {
    __m256 af = _mm256_loadu_ps(a32 + i);
    __m128i q8  = _mm_loadl_epi64((const __m128i*)(q8_64 + i));
    __m256i q32 = _mm256_cvtepi8_epi32(q8);
    __m256  qf  = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), vscale);
    acc = _mm256_fmadd_ps(af, qf, acc);
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float out; _mm_store_ss(&out, sum);
  return out;
}
#endif

// Multithreaded, Q8-expanded weights: B_packed_q8 is int8, 64 bytes per group
void qgemm_int4_fp16_q8_mt(
  const uint16_t* A_fp16, int lda,
  const int8_t*   B_packed_q8, const uint16_t* B_scales, int /*ldb_q8*/,
  uint16_t*       C_fp16, int ldc,
  int M, int N, int K, int group_size,
  int num_threads)
{
  if (num_threads <= 0) {
    num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads <= 0) num_threads = 4;
  }
  num_threads = std::max(1, std::min(num_threads, N));

  const int groups = (K + group_size - 1) / group_size;
  const int bytes_per_group = group_size; // 64 int8

  const bool can_use_avx2 =
  #if defined(__AVX2__) && defined(__F16C__)
    (group_size == 64);
  #else
    false;
  #endif

  parallel_for(0, N, num_threads, [&](int n0, int n1){
    alignas(32) float a32[64];
    for (int n = n0; n < n1; ++n) {
      const int8_t*   Bcol = B_packed_q8 + n * (groups * bytes_per_group);
      const uint16_t* Sc   = B_scales     + n * groups;

      for (int m=0; m<M; ++m) {
        const uint16_t* Arow = A_fp16 + m*lda;
        float acc = 0.0f;
        int k = 0;

        for (int g=0; g<groups; ++g) {
          const float s = fp16_to_fp32(Sc[g]);
          const int8_t* group_ptr = Bcol + g * bytes_per_group;

          if (can_use_avx2 && (k + 64) <= K) {
            #if defined(__AVX2__) && defined(__F16C__)
            convert_a64_fp16_to_fp32(Arow + k, a32);
            acc += dot64_from_a32_q8_avx2(a32, group_ptr, s);
            k += 64;
            continue;
            #endif
          }

          // scalar tail
          for (int i=0; i<bytes_per_group; ++i) {
            if ((k+i) < K) {
              acc += fp16_to_fp32(Arow[k+i]) * (s * (float)group_ptr[i]);
            }
          }
          k += 64;
        }
        C_fp16[m*ldc + n] = fp16_from_fp32(acc);
      }
    }
  });
}


void qgemm_int4_fp16_tiled_mt_mblocked(
  const uint16_t* A_fp16, int lda,
  const uint8_t*  B_packed, const uint16_t* B_scales, int /*ldb_packed*/,
  uint16_t*       C_fp16, int ldc,
  int M, int N, int K, int group_size,
  int num_threads, int nc_tile, int m_tile)
{
  if (num_threads <= 0) {
    num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads <= 0) num_threads = 4;
  }
  if (nc_tile <= 0) nc_tile = 8;
  if (m_tile <= 0)  m_tile = 4;

  const int groups = (K + group_size - 1) / group_size;
  const int bytes_per_group = group_size / 2;

  const int num_tiles = (N + nc_tile - 1) / nc_tile;
  num_threads = std::max(1, std::min(num_threads, num_tiles));

  parallel_for(0, num_tiles, num_threads, [&](int t0, int t1){
    alignas(32) float a32[64];
    for (int t=t0; t<t1; ++t) {
      const int n_begin = t * nc_tile;
      const int n_end   = std::min(N, n_begin + nc_tile);
      const int cols    = n_end - n_begin;

      for (int m0=0; m0<M; m0+=m_tile) {
        const int mr = std::min(m_tile, M - m0);
        float acc[32][8]; // supports up to nc_tile<=32 and m_tile<=8
        for (int i=0;i<cols;++i) for(int r=0;r<mr;++r) acc[i][r]=0.f;

        for (int g=0, k=0; g<groups; ++g, k+=64) {
          for (int r=0;r<mr;++r) {
          #if defined(__AVX2__) && defined(__F16C__)
            if (K - k >= 64) convert_a64_fp16_to_fp32(A_fp16 + (m0+r)*lda + k, a32);
          #endif
            for (int i=0;i<cols;++i) {
              const int n = n_begin + i;
              const uint8_t*  group_ptr = B_packed + n * (groups * bytes_per_group) + g * bytes_per_group;
              const float     s = fp16_to_fp32(B_scales[n * groups + g]);

            #if defined(__AVX2__) && defined(__F16C__)
              if (K - k >= 64) { acc[i][r] += dot64_from_a32_avx2(a32, group_ptr, s); continue; }
            #endif
              int kk=k;
              for (int b=0;b<bytes_per_group;++b){
                uint8_t v=group_ptr[b];
                int q0=((v>>4)&0xF)-8, q1=((v)&0xF)-8;
                if (kk   < K) acc[i][r] += fp16_to_fp32(A_fp16[(m0+r)*lda + kk   ]) * (s*(float)q0);
                if (kk+1 < K) acc[i][r] += fp16_to_fp32(A_fp16[(m0+r)*lda + kk+1 ]) * (s*(float)q1);
                kk+=2;
              }
            }
          }
        }
        for (int i=0;i<cols;++i) for(int r=0;r<mr;++r)
          C_fp16[(m0+r)*ldc + (n_begin+i)] = fp16_from_fp32(acc[i][r]);
      }
    }
  });
}
