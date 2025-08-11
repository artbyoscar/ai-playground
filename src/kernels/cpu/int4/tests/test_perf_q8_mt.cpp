#include "../qgemm_int4.h"
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>

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
  int M=256,N=256,K=2048,it=3, threads=0;
  const char* json_out = nullptr;

  for (int i=1;i<argc;++i){
    if(!std::strcmp(argv[i],"--M") && i+1<argc) M=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--N") && i+1<argc) N=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--K") && i+1<argc) K=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--it")&& i+1<argc) it=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--threads")&& i+1<argc) threads=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--json")&& i+1<argc) json_out=argv[++i];
  }

  // Random A (fp16) and Q8 B
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f,1.f);

  std::vector<uint16_t> Ah((size_t)M*K);
  for (int i=0;i<M*K;++i) Ah[i]=f2h(dist(rng));

  // Make a simple Q8 B + fp16 scales (reuse int4 layout idea)
  const int G = 64;
  const int groups=(K+G-1)/G;
  std::vector<int8_t>   Bq((size_t)K*N);
  std::vector<uint16_t> Sc((size_t)N*groups);
  // naïve per-group scale = 1.0 for now, and pack signed [-127,127]
  for (auto &v:Bq){ v = (int8_t)std::round(dist(rng)*100); }
  for (auto &s:Sc){ s = f2h(1.0f); }

  auto bench = [&](auto fn, const char* name)->double{
    std::vector<uint16_t> Ch((size_t)M*N);
    fn(Ah.data(), K, (const int8_t*)Bq.data(), (const uint16_t*)Sc.data(), N, Ch.data(), N, M,N,K,G, (threads>0?threads:8));
    double best=1e100, gflop=(2.0*M*N*K)/1e9;
    for(int i=0;i<it;++i){
      auto t0=std::chrono::high_resolution_clock::now();
      fn(Ah.data(), K, (const int8_t*)Bq.data(), (const uint16_t*)Sc.data(), N, Ch.data(), N, M,N,K,G, (threads>0?threads:8));
      best = std::min(best, ms_since(t0));
    }
    std::printf("  %-6s : %.3f ms  (%.2f GFLOP/s)\n", name, best, gflop/(best/1000.0));
    return best;
  };

  std::printf("Q8 MT perf: M=%d N=%d K=%d it=%d threads=%d\n",M,N,K,it,threads);
  double mt = bench([](auto*Ah,int lda,auto*Bq,auto*Sc,int ldb,auto*Ch,int ldc,int M,int N,int K,int G,int thr){
    qgemm_int4_fp16_q8_mt(Ah, lda, (const int8_t*)Bq, (const uint16_t*)Sc, ldb, Ch, ldc, M,N,K,G, thr);
  },"mt");

  if (json_out){
    FILE* f = std::fopen(json_out,"wb");
    if (f){
      std::fprintf(f, "{\"M\":%d,\"N\":%d,\"K\":%d,\"it\":%d,\"mt_ms\":%.4f}\n", M,N,K,it, mt);
      std::fclose(f);
    }
  }
  return 0;
}
