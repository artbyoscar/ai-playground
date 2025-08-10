#include "../qgemm_int4.h"
#include "../pack_int4.h"

#include <random>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

// correct scalar fp16 <-> fp32 (matches kernel fallback)
static inline float h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -1;
      do { mant <<= 1; ++e; } while ((mant & 0x400) == 0);
      mant &= 0x3FF;
      uint32_t exp_f = (127 - 15 - e);
      bits = sign | (exp_f << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    uint32_t exp_f = exp - 15 + 127;
    bits = sign | (exp_f << 23) | (mant << 13);
  }
  float out; std::memcpy(&out, &bits, sizeof(out));
  return out;
}
static inline uint16_t f2h(float f) {
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
    return (uint16_t)(sign | 0x7BFFu);
  } else {
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) half += 1u; // RNE
    return (uint16_t)half;
  }
}

// slow, blasless reference
static void gemm_f32_ref(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m=0; m<M; ++m) for (int n=0; n<N; ++n) {
    float acc = 0.f;
    for (int k=0; k<K; ++k) acc += A[m*K + k] * B[k*N + n];
    C[m*N + n] = acc;
  }
}

struct Stats { double max_abs, mean_abs, rel_err; };
static Stats compare(const std::vector<float>& X, const std::vector<float>& Y) {
  double max_abs=0, sum_abs=0, den=0;
  for (size_t i=0;i<X.size();++i) {
    double d = std::fabs(double(X[i])-double(Y[i]));
    max_abs = std::max(max_abs, d);
    sum_abs += d;
    den += std::fabs(double(Y[i]));
  }
  double mean_abs = sum_abs / std::max<size_t>(1, X.size());
  double rel_err  = (den>1e-12)? (sum_abs/den) : max_abs;
  return {max_abs, mean_abs, rel_err};
}

int main(int argc, char** argv) {
  double threshold = 5e-2;
  for (int i=1;i<argc;++i)
    if (!std::strcmp(argv[i],"--threshold") && i+1<argc) threshold = std::atof(argv[++i]);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  const int SHAPES[][3] = {
    {32, 32, 256},
    {64, 64, 512},
    {64, 64, 4096},
    {48, 48, 1000},
  };

  const int G = 64;
  bool ok_all = true;

  for (auto& s : SHAPES) {
    int M=s[0], N=s[1], K=s[2];
    std::printf("== Shape M=%d N=%d K=%d\n", M,N,K);

    std::vector<float> A(M*K), B(K*N), C_ref(M*N), C0(M*N), C1(M*N), C2(M*N); 
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    gemm_f32_ref(A.data(), B.data(), C_ref.data(), M,N,K);

    // pack B per-column along K (matches kernel)
    const int groups = (K + G - 1) / G;
    const int bytes_per_group = G / 2;
    std::vector<uint8_t>  B_packed;  B_packed.reserve(size_t(N)*groups*bytes_per_group);
    std::vector<uint16_t> B_scales;  B_scales.reserve(size_t(N)*groups);

    std::vector<float> col((size_t)K);
    for (int n=0; n<N; ++n) {
      for (int k=0; k<K; ++k) col[(size_t)k] = B[(size_t)k*N + n];
      auto pk = q4edge_pack_rowmajor_f32(col.data(), 1, K, G);
      B_scales.insert(B_scales.end(), pk.scales.begin(), pk.scales.end());
      B_packed.insert(B_packed.end(), pk.data.begin(), pk.data.end());
    }

    // A -> fp16
    std::vector<uint16_t> A_h(M*K), C_h0(M*N), C_h1(M*N), C_h2(M*N);
    for (int i=0;i<M*K;++i) A_h[(size_t)i] = f2h(A[(size_t)i]);

    // 1) single-thread
    qgemm_int4_fp16(A_h.data(), K, B_packed.data(), B_scales.data(), 0, C_h0.data(), N, M,N,K, G);
    for (int i=0;i<M*N;++i) C0[(size_t)i] = h2f(C_h0[(size_t)i]);

    // 2) MT
    qgemm_int4_fp16_mt(A_h.data(), K, B_packed.data(), B_scales.data(), 0, C_h1.data(), N, M,N,K, G, std::min(N, 8));
    for (int i=0;i<M*N;++i) C1[(size_t)i] = h2f(C_h1[(size_t)i]);

    // 3) tiled MT
    qgemm_int4_fp16_tiled_mt(A_h.data(), K, B_packed.data(), B_scales.data(), 0, C_h2.data(), N, M,N,K, G, std::min(N, 8), 8);
    for (int i=0;i<M*N;++i) C2[(size_t)i] = h2f(C_h2[(size_t)i]);

    auto s0 = compare(C0, C_ref);
    auto s1 = compare(C1, C_ref);
    auto s2 = compare(C2, C_ref);

    std::printf("  st:  max=%.4e  mean=%.4e  rel=%.4e\n", s0.max_abs, s0.mean_abs, s0.rel_err);
    std::printf("  mt:  max=%.4e  mean=%.4e  rel=%.4e\n", s1.max_abs, s1.mean_abs, s1.rel_err);
    std::printf("  tmt: max=%.4e  mean=%.4e  rel=%.4e\n", s2.max_abs, s2.mean_abs, s2.rel_err);

    bool ok = (s0.rel_err <= threshold) && (s1.rel_err <= threshold) && (s2.rel_err <= threshold);
    ok_all = ok_all && ok;
    if (!ok) std::printf("  FAIL: rel_err exceeded threshold %.2e\n", threshold);
    else     std::printf("  PASS\n");
  }

  return ok_all ? 0 : 1;
}
