#include "../qgemm_int4.h"
#include "../pack_loader.h"
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>

static double ms_since(std::chrono::high_resolution_clock::time_point t0){
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now()-t0).count();
}
static inline uint16_t f2h(float f){
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
}

int main(int argc, char** argv){
  const char* prefix = "pack/q4edge";
  int M = 256, it = 5;
  for(int i=1;i<argc;++i){
    if(!std::strcmp(argv[i],"--packed") && i+1<argc) prefix = argv[++i];
    else if(!std::strcmp(argv[i],"--M") && i+1<argc) M = std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--it")&& i+1<argc) it= std::atoi(argv[++i]);
  }

  auto pb = load_q4edge_packed(prefix);
  int K = pb.K, N = pb.N, G = pb.group;
  printf("Loaded packed B: K=%d N=%d group=%d\n", K,N,G);

  // Random A (M x K) -> fp16
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f,1.f);
  std::vector<uint16_t> Ah((size_t)M*K);
  for(int i=0;i<M*K;++i) Ah[i] = f2h(dist(rng));

  auto bench = [&](auto fn, const char* name)->double{
    std::vector<uint16_t> Ch((size_t)M*N);
    // warmup
    fn(Ah.data(), K, pb.data.get(), pb.scales.get(), 0, Ch.data(), N, M,N,K,G);
    double best=1e100, gflop = (2.0 * M * N * K) / 1e9;
    for(int i=0;i<it;++i){
      auto t1=std::chrono::high_resolution_clock::now();
      fn(Ah.data(), K, pb.data.get(), pb.scales.get(), 0, Ch.data(), N, M,N,K,G);
      double ms = ms_since(t1);
      best = std::min(best, ms);
    }
    printf("  %-6s : %.3f ms  (%.2f GFLOP/s)\n", name, best, (2.0*M*N*K/1e9)/(best/1000.0));
    return best;
  };

  printf("Perf vs packed: M=%d N=%d K=%d it=%d\n",M,N,K,it);
  bench([](auto*Ah,int lda,auto*Bp,auto*Sc,int ldb,auto*Ch,int ldc,int M,int N,int K,int G){
    qgemm_int4_fp16(Ah, lda, (const uint8_t*)Bp, (const uint16_t*)Sc, ldb, Ch, ldc, M,N,K,G);
  },"st");
  bench([](auto*Ah,int lda,auto*Bp,auto*Sc,int ldb,auto*Ch,int ldc,int M,int N,int K,int G){
    qgemm_int4_fp16_mt(Ah, lda, (const uint8_t*)Bp, (const uint16_t*)Sc, ldb, Ch, ldc, M,N,K,G, std::min(N, 8));
  },"mt");
  bench([](auto*Ah,int lda,auto*Bp,auto*Sc,int ldb,auto*Ch,int ldc,int M,int N,int K,int G){
    qgemm_int4_fp16_tiled_mt(Ah, lda, (const uint8_t*)Bp, (const uint16_t*)Sc, ldb, Ch, ldc, M,N,K,G, std::min(N, 8), 8);
  },"tmt");
  return 0;
}