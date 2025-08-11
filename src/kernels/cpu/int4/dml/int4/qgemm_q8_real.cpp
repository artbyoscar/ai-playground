// qgemm_q8_real.cpp
#include "qgemm_int4.h"
#include <immintrin.h>
#include <thread>
#include <vector>
#include <cstring>
#include <algorithm>

// Helper functions for FP16 conversion
static inline float h2f(uint16_t h) {
    uint16_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t out;
    if (e == 0) {
        if (m == 0) out = s << 31;
        else {
            e = 1;
            while ((m & 1024) == 0) { m <<= 1; e--; }
            m &= 1023;
            out = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7F800000 | (m << 13);
    } else {
        out = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

static inline uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t t = mant >> (1 - exp + 13);
        if ((mant >> (1 - exp + 12)) & 1u) t += 1u;
        return (uint16_t)(sign | t);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7BFFu);
    }
    
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) half += 1u;
    return (uint16_t)half;
}

// Q8 kernel - single threaded version
static void qgemm_q8_fp16_st(
    const uint16_t* A, int lda,
    const int8_t* B_q8,
    const uint16_t* scales, int ldb,
    uint16_t* C, int ldc,
    int M, int N, int K, int group_size) {
    
    const int groups_per_col = (K + group_size - 1) / group_size;
    
    // Process each output element
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            
            // Process K dimension in groups
            for (int g = 0; g < groups_per_col; ++g) {
                int k_start = g * group_size;
                int k_end = std::min(k_start + group_size, K);
                
                // Get scale for this group
                float scale = h2f(scales[n * groups_per_col + g]);
                
                // Process this group with AVX2
                int k = k_start;
                
                // AVX2 path for chunks of 8
                __m256 acc_vec = _mm256_setzero_ps();
                for (; k + 7 < k_end; k += 8) {
                    // Load 8 A values (fp16 -> fp32)
                    __m128i a_half = _mm_loadu_si128((__m128i*)&A[m * lda + k]);
                    __m256 a_vec = _mm256_cvtph_ps(a_half);
                    
                    // Load 8 B values (int8 -> fp32)
                    __m128i b_bytes = _mm_loadl_epi64((__m128i*)&B_q8[n * K + k]);
                    __m128i b_words = _mm_cvtepi8_epi16(b_bytes);
                    __m128i b_lo = _mm_cvtepi16_epi32(b_words);
                    __m128i b_hi = _mm_cvtepi16_epi32(_mm_srli_si128(b_words, 8));
                    __m256i b_int = _mm256_set_m128i(b_hi, b_lo);
                    __m256 b_vec = _mm256_cvtepi32_ps(b_int);
                    
                    // Apply scale and accumulate
                    b_vec = _mm256_mul_ps(b_vec, _mm256_set1_ps(scale));
                    acc_vec = _mm256_fmadd_ps(a_vec, b_vec, acc_vec);
                }
                
                // Horizontal sum
                __m128 acc_hi = _mm256_extractf128_ps(acc_vec, 1);
                __m128 acc_lo = _mm256_castps256_ps128(acc_vec);
                __m128 acc_sum = _mm_add_ps(acc_hi, acc_lo);
                acc_sum = _mm_hadd_ps(acc_sum, acc_sum);
                acc_sum = _mm_hadd_ps(acc_sum, acc_sum);
                acc += _mm_cvtss_f32(acc_sum);
                
                // Scalar cleanup
                for (; k < k_end; ++k) {
                    float a_val = h2f(A[m * lda + k]);
                    float b_val = (float)B_q8[n * K + k] * scale;
                    acc += a_val * b_val;
                }
            }
            
            C[m * ldc + n] = f2h(acc);
        }
    }
}

// Q8 kernel - multi-threaded version
void qgemm_int4_fp16_q8_mt(
    const uint16_t* A, int lda,
    const int8_t* B_q8,
    const uint16_t* scales, int ldb,
    uint16_t* C, int ldc,
    int M, int N, int K, int group_size, int num_threads) {
    
    if (num_threads <= 1 || N < num_threads) {
        qgemm_q8_fp16_st(A, lda, B_q8, scales, ldb, C, ldc, M, N, K, group_size);
        return;
    }
    
    // Partition work across N dimension
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    int cols_per_thread = (N + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        int n_start = t * cols_per_thread;
        int n_end = std::min(n_start + cols_per_thread, N);
        
        if (n_start >= N) break;
        
        threads.emplace_back([=]() {
            const int groups_per_col = (K + group_size - 1) / group_size;
            
            for (int m = 0; m < M; ++m) {
                for (int n = n_start; n < n_end; ++n) {
                    float acc = 0.0f;
                    
                    for (int g = 0; g < groups_per_col; ++g) {
                        int k_start = g * group_size;
                        int k_end = std::min(k_start + group_size, K);
                        
                        float scale = h2f(scales[n * groups_per_col + g]);
                        
                        int k = k_start;
                        __m256 acc_vec = _mm256_setzero_ps();
                        
                        // AVX2 processing
                        for (; k + 7 < k_end; k += 8) {
                            __m128i a_half = _mm_loadu_si128((__m128i*)&A[m * lda + k]);
                            __m256 a_vec = _mm256_cvtph_ps(a_half);
                            
                            __m128i b_bytes = _mm_loadl_epi64((__m128i*)&B_q8[n * K + k]);
                            __m128i b_words = _mm_cvtepi8_epi16(b_bytes);
                            __m128i b_lo = _mm_cvtepi16_epi32(b_words);
                            __m128i b_hi = _mm_cvtepi16_epi32(_mm_srli_si128(b_words, 8));
                            __m256i b_int = _mm256_set_m128i(b_hi, b_lo);
                            __m256 b_vec = _mm256_cvtepi32_ps(b_int);
                            
                            b_vec = _mm256_mul_ps(b_vec, _mm256_set1_ps(scale));
                            acc_vec = _mm256_fmadd_ps(a_vec, b_vec, acc_vec);
                        }
                        
                        // Reduce
                        __m128 acc_hi = _mm256_extractf128_ps(acc_vec, 1);
                        __m128 acc_lo = _mm256_castps256_ps128(acc_vec);
                        __m128 acc_sum = _mm_add_ps(acc_hi, acc_lo);
                        acc_sum = _mm_hadd_ps(acc_sum, acc_sum);
                        acc_sum = _mm_hadd_ps(acc_sum, acc_sum);
                        acc += _mm_cvtss_f32(acc_sum);
                        
                        // Scalar cleanup
                        for (; k < k_end; ++k) {
                            float a_val = h2f(A[m * lda + k]);
                            float b_val = (float)B_q8[n * K + k] * scale;
                            acc += a_val * b_val;
                        }
                    }
                    
                    C[m * ldc + n] = f2h(acc);
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
}