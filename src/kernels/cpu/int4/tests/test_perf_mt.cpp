#include "../qgemm_int4.h"
#include "../pack_int4.h"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <thread>


static inline uint16_t f2h(float f) {
  uint16_t s = (f < 0) ? 0x8000 : 0;
  float a = std::fabs(f); if (a==0) return s;
  int e; float m = std::frexp(a, &e);
  int E = std::clamp(e + 14, -15, 16) + 15;
  uint16_t M = (uint16_t)std::clamp(int((m*2 - 1)*(1<<10)),0,(1<<10)-1);
  return s | (uint16_t(E)<<10) | M;
}

int main() {
  const int M=64, N=64, K=4096, G=64;
  std::mt19937 rng(321);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> A_f32(size_t(M)*K);
  std::vector<float> B_f32(size_t(K)*N);
  for (auto& x: A_f32) x = dist(rng);
  for (auto& x: B_f32) x = dist(rng);

  // pack B by columns
  const int groups = (K+G-1)/G;
  std::vector<uint8_t>  B_packed; B_packed.reserve(size_t(N)*K/2);
  std::vector<uint16_t> B_scales; B_scales.reserve(size_t(N)*groups);
  std::vector<float> col(static_cast<size_t>(K));
  for (int n=0; n<N; ++n) {
    for (int k=0; k<K; ++k) col[k] = B_f32[k*N + n];
    auto p = q4edge_pack_rowmajor_f32(col.data(), 1, K, G);
    for (int g=0; g<groups; ++g) B_scales.push_back(p.scales[g]);
    B_packed.insert(B_packed.end(), p.data.begin(), p.data.end());
  }

  // A in fp16
  std::vector<uint16_t> A_fp16(size_t(M)*K), C_fp16(size_t(M)*N);
  for (size_t i=0;i<A_f32.size();++i) A_fp16[i] = f2h(A_f32[i]);

  // run for T in {1,2,4,8,16}
  int hw = (int)std::thread::hardware_concurrency();
  std::vector<int> Ts = {1,2,4,8,16};
  for (auto& t : Ts) if (t > hw) { Ts.resize(&t - Ts.data()); break; }

  for (int T : Ts) {
    // warmup
    for (int w=0; w<3; ++w)
      qgemm_int4_fp16_mt(A_fp16.data(), K, B_packed.data(), B_scales.data(), 0,
                         C_fp16.data(), N, M, N, K, G, T);

    // timed
    auto t0 = std::chrono::high_resolution_clock::now();
    const int iters = 30;
    for (int it=0; it<iters; ++it)
      qgemm_int4_fp16_mt(A_fp16.data(), K, B_packed.data(), B_scales.data(), 0,
                         C_fp16.data(), N, M, N, K, G, T);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count() / iters;

    double gflops = (2.0 * M * N * (double)K) / 1e9;
    double gflops_per_s = gflops / (ms/1000.0);

    std::cout << "Threads="<<T<<"  avg="<<ms<<" ms  ~"<<gflops_per_s<<" GFLOP/s\n";
  }
  return 0;
}
