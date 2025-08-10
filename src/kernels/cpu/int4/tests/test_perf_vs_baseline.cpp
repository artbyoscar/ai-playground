#include "../qgemm_int4.h"
#include "../pack_int4.h"

#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>

// forward decl for the optional M-blocked kernel we added in qgemm_int4.cpp
void qgemm_int4_fp16_tiled_mt_mblocked(
  const uint16_t* A_fp16, int lda,
  const uint8_t*  B_packed, const uint16_t* B_scales, int ldb_packed,
  uint16_t*       C_fp16, int ldc,
  int M, int N, int K, int group_size,
  int num_threads, int nc_tile, int m_tile);

static void gemm_f32_ref(const float* A,const float* B,float* C,int M,int N,int K){
  for(int m=0;m<M;++m){
    for(int n=0;n<N;++n){
      float acc=0.f;
      for(int k=0;k<K;++k) acc += A[m*K+k]*B[k*N+n];
      C[m*N+n]=acc;
    }
  }
}
static double ms_since(std::chrono::high_resolution_clock::time_point t0){
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now()-t0).count();
}

int main(int argc,char**argv){
  int M=256,N=256,K=2048, it=5;
  for(int i=1;i<argc;++i){
    if(!std::strcmp(argv[i],"--M") && i+1<argc) M=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--N") && i+1<argc) N=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--K") && i+1<argc) K=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--it")&& i+1<argc) it=std::atoi(argv[++i]);
  }

  std::printf("Perf vs baseline: M=%d N=%d K=%d it=%d\n",M,N,K,it);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f,1.f);

  std::vector<float> A(M*K), B(K*N), C_ref(M*N,0.f);
  for(auto&x:A) x=dist(rng);
  for(auto&x:B) x=dist(rng);

  // FP32 reference
  auto t0 = std::chrono::high_resolution_clock::now();
  gemm_f32_ref(A.data(),B.data(),C_ref.data(),M,N,K);
  double t_ref = ms_since(t0);
  double gflop = (2.0 * M * N * K) / 1e9;
  std::printf("  FP32 ref:  %.3f ms  (%.2f GFLOP/s)\n", t_ref, gflop/(t_ref/1000.0));

  // Pack B colwise (group=64)
  const int G=64;
  const int groups=(K+G-1)/G;
  const int bytes_per_group=G/2;
  std::vector<uint8_t>  Bp; Bp.reserve((size_t)N*groups*bytes_per_group);
  std::vector<uint16_t> Sc; Sc.reserve((size_t)N*groups);
  std::vector<float> col(K);
  for(int n=0;n<N;++n){
    for(int k=0;k<K;++k) col[k]=B[(size_t)k*N+n];
    auto pk = q4edge_pack_rowmajor_f32(col.data(),1,K,G);
    Sc.insert(Sc.end(), pk.scales.begin(), pk.scales.end());
    Bp.insert(Bp.end(), pk.data.begin(), pk.data.end());
  }

  // A -> fp16
  auto f2h = [](float f)->uint16_t{
    uint32_t x; std::memcpy(&x,&f,sizeof(x));
    uint32_t sign=(x>>16)&0x8000u;
    int32_t  exp =(int32_t)((x>>23)&0xFF)-127+15;
    uint32_t mant=x&0x7FFFFFu;
    if(exp<=0){ if(exp<-10) return (uint16_t)sign; mant|=0x800000u;
      uint32_t t = mant >> (1-exp+13);
      if((mant>>(1-exp+12))&1u) t+=1u; return (uint16_t)(sign|t);
    } else if(exp>=31){ return (uint16_t)(sign|0x7BFFu); }
    uint32_t half=sign|((uint32_t)exp<<10)|(mant>>13);
    if(mant&0x00001000u) half+=1u; return (uint16_t)half;
  };
  std::vector<uint16_t> Ah(M*K);
  for(int i=0;i<M*K;++i) Ah[i]=f2h(A[i]);

  // Bench helper
  auto bench = [&](auto fn, const char* name)->double{
    std::vector<uint16_t> Ch(M*N);
    fn(Ah.data(), K, Bp.data(), Sc.data(), 0, Ch.data(), N, M,N,K, G); // warmup
    double best=1e100;
    for(int i=0;i<it;++i){
      auto t1=std::chrono::high_resolution_clock::now();
      fn(Ah.data(), K, Bp.data(), Sc.data(), 0, Ch.data(), N, M,N,K, G);
      best = std::min(best, ms_since(t1));
    }
    std::printf("  %-6s : %.3f ms  (%.2f GFLOP/s)  speedup vs FP32: %.2fx\n",
                name, best, gflop/(best/1000.0), t_ref/best);
    return best;
  };

  bench([](auto*Ah,int lda,auto*Bp,auto*Sc,int ldb,auto*Ch,int ldc,int M,int N,int K,int G){
    qgemm_int4_fp16(Ah, lda, (const uint8_t*)Bp, (const uint16_t*)Sc, ldb, Ch, ldc, M,N,K,G);
  },"st");

  bench([](auto*Ah,int lda,auto*Bp,auto*Sc,int ldb,auto*Ch,int ldc,int M,int N,int K,int G){
    qgemm_int4_fp16_mt(Ah, lda, (const uint8_t*)Bp, (const uint16_t*)Sc, ldb, Ch, ldc, M,N,K,G, std::min(N, 8));
  },"mt");

  bench([](auto*Ah,int lda,auto*Bp,auto*Sc,int ldb,auto*Ch,int ldc,int M,int N,int K,int G){
    qgemm_int4_fp16_tiled_mt(Ah, lda, (const uint8_t*)Bp, (const uint16_t*)Sc, ldb, Ch, ldc, M,N,K,G,
                             std::min(N, 8), 8);
  },"tmt");

  // new: tmt + m-blocking
  bench([](auto*Ah,int lda,auto*Bp,auto*Sc,int ldb,auto*Ch,int ldc,int M,int N,int K,int G){
    qgemm_int4_fp16_tiled_mt_mblocked(Ah, lda, (const uint8_t*)Bp, (const uint16_t*)Sc, ldb,
                                      Ch, ldc, M,N,K,G, std::min(N, 8), 8, 4);
  },"tmt+m");

  return 0;
}
