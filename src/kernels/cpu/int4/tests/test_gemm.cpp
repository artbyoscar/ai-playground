#include "../qgemm_int4.h"
#include "../pack_int4.h"
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

static inline uint16_t f2h(float f) {
  uint16_t s = (f < 0) ? 0x8000 : 0;
  float a = std::fabs(f); if (a==0) return s;
  int e; float m = std::frexp(a, &e);
  int E = std::clamp(e + 14, -15, 16) + 15;
  uint16_t M = (uint16_t)std::clamp(int((m*2 - 1)*(1<<10)),0,(1<<10)-1);
  return s | (uint16_t(E)<<10) | M;
}

int main() {
  const int M=32, K=256, N=32, G=64;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> A_f32(size_t(M)*K);
  std::vector<float> B_f32(size_t(K)*N);
  for (auto& x : A_f32) x = dist(rng);
  for (auto& x : B_f32) x = dist(rng);

  // Pack each column of B using the row packer (rows=1, cols=K)
  std::vector<uint8_t>  B_packed; B_packed.reserve(size_t(N)*K/2);
  std::vector<uint16_t> B_scales; B_scales.reserve(size_t(N)*(K+G-1)/G);
  int groups = (K+G-1)/G;

  std::vector<float> col(static_cast<size_t>(K));
  for (int n=0; n<N; ++n) {
    for (int k=0; k<K; ++k) col[k] = B_f32[k*N + n];
    auto p = q4edge_pack_rowmajor_f32(col.data(), 1, K, G);
    // append scales then bytes for that column
    for (int g=0; g<groups; ++g) B_scales.push_back(p.scales[g]);
    B_packed.insert(B_packed.end(), p.data.begin(), p.data.end());
  }

  // Convert A to fp16
  std::vector<uint16_t> A_fp16(size_t(M)*K), C_fp16(size_t(M)*N, 0);
  for (size_t i=0;i<A_f32.size();++i) A_fp16[i] = f2h(A_f32[i]);

  qgemm_int4_fp16(A_fp16.data(), K, B_packed.data(), B_scales.data(), 0,
                  C_fp16.data(), N, M, N, K, G);

  std::cout << "OK: ran qgemm_int4_fp16 with M="<<M<<" K="<<K<<" N="<<N<<"\n";
  return 0;
}
