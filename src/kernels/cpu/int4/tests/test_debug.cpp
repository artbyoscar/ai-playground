// test_debug.cpp - Simple diagnostic test to identify the issue
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>

// Include the headers
#include "qgemm_int4.h"

// Copy the fp16 conversion functions from qgemm_int4.cpp for testing
#if defined(__F16C__)
  #include <immintrin.h>
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
  // Scalar fallback
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
    float out; 
    std::memcpy(&out, &bits, sizeof(out));
    return out;
  }
  
  static inline uint16_t fp16_from_fp32(float f) {
    uint32_t x; 
    std::memcpy(&x, &f, sizeof(x));
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

void test_fp16_conversion() {
    std::cout << "\n=== Testing FP16 Conversion ===" << std::endl;
    
    float test_values[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 100.0f, -100.0f, 
                          0.1f, 0.01f, 65504.0f, -65504.0f};
    
    for (float val : test_values) {
        uint16_t h = fp16_from_fp32(val);
        float recovered = fp16_to_fp32(h);
        std::cout << "f32: " << std::setw(12) << val 
                  << " -> fp16: 0x" << std::hex << std::setw(4) << std::setfill('0') << h
                  << std::dec << " (" << h << ")"
                  << " -> f32: " << std::setw(12) << recovered
                  << " | diff: " << std::scientific << std::abs(val - recovered)
                  << std::endl;
    }
}

void test_simple_gemm() {
    std::cout << "\n=== Testing Simple 2x2 GEMM ===" << std::endl;
    
    // Very simple test: 2x2 matrices
    const int M = 2, N = 2, K = 64;  // K=64 for one group
    const int group_size = 64;
    
    // Create simple input A (all ones)
    std::vector<uint16_t> A(M * K);
    for (int i = 0; i < M * K; i++) {
        A[i] = fp16_from_fp32(1.0f);
    }
    
    // Create simple quantized B (alternating -8 and 7)
    // B should be packed as 4-bit values
    const int groups = 1;
    const int bytes_per_group = group_size / 2; // 32 bytes for 64 values
    std::vector<uint8_t> B_packed(N * groups * bytes_per_group);
    
    // Pack values: each byte holds two 4-bit values
    // We'll use 0 (which maps to -8) and 15 (which maps to 7)
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < bytes_per_group; b++) {
            // Pack two values: 0 (-8) in high nibble, 15 (+7) in low nibble
            B_packed[n * bytes_per_group + b] = 0x0F; // binary: 0000 1111
        }
    }
    
    // Scales (just 1.0 for simplicity)
    std::vector<uint16_t> B_scales(N * groups);
    for (int i = 0; i < N * groups; i++) {
        B_scales[i] = fp16_from_fp32(1.0f);
    }
    
    // Output
    std::vector<uint16_t> C(M * N, 0);
    
    // Run the kernel
    qgemm_int4_fp16(A.data(), K, B_packed.data(), B_scales.data(), 0, 
                    C.data(), N, M, N, K, group_size);
    
    // Check results
    std::cout << "Input A (all ones): " << std::endl;
    for (int m = 0; m < M; m++) {
        std::cout << "  Row " << m << ": ";
        for (int k = 0; k < std::min(8, K); k++) {
            std::cout << fp16_to_fp32(A[m * K + k]) << " ";
        }
        std::cout << "... (all 1.0)" << std::endl;
    }
    
    std::cout << "\nQuantized B pattern: alternating -8 and +7" << std::endl;
    std::cout << "Scale: 1.0" << std::endl;
    std::cout << "Expected dot product per element: 32*(-8) + 32*(7) = -256 + 224 = -32" << std::endl;
    
    std::cout << "\nOutput C:" << std::endl;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float val = fp16_to_fp32(C[m * N + n]);
            std::cout << "  C[" << m << "," << n << "] = " << val 
                      << " (fp16: 0x" << std::hex << C[m * N + n] << std::dec << ")"
                      << std::endl;
        }
    }
}

void test_unpack() {
    std::cout << "\n=== Testing 4-bit Unpacking ===" << std::endl;
    
    uint8_t packed_bytes[] = {0x0F, 0x8F, 0x78, 0xFF, 0x00};
    
    for (uint8_t byte : packed_bytes) {
        int q0 = (byte >> 4) & 0xF;
        int q1 = (byte) & 0xF;
        int v0 = q0 - 8;  // Map 0..15 to -8..7
        int v1 = q1 - 8;
        
        std::cout << "Byte 0x" << std::hex << std::setw(2) << std::setfill('0') << (int)byte 
                  << std::dec
                  << " -> nibbles: " << q0 << "," << q1
                  << " -> values: " << v0 << "," << v1 
                  << std::endl;
    }
}

int main() {
    std::cout << "=== QGEMM Debug Test ===" << std::endl;
    
    test_fp16_conversion();
    test_unpack();
    test_simple_gemm();
    
    return 0;
}