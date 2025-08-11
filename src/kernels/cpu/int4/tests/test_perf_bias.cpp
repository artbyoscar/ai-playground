// Replace the entire test_perf_bias.cpp with this updated version:
#include "../qgemm_int4.h"
#include "../pack_int4.h"
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

static inline float h2f(uint16_t h){
  uint16_t s=(h>>15)&1, e=(h>>10)&31, m=h&1023;
  uint32_t out;
  if(e==0){ if(m==0) out=s<<31; else{
    e=1; while((m&1024)==0){ m<<=1; e--; } m&=1023;
    out=(s<<31)|((e+112)<<23)|(m<<13);
  }} else if(e==31){ out=(s<<31)|0x7F800000|(m<<13); }
  else{ out=(s<<31)|((e+112)<<23)|(m<<13); }
  float f; std::memcpy(&f,&out,4); return f;
}

int main(int argc,char**argv){
  int M=256,N=256,K=2048,it=3, use_bias=1, relu=1, use_fused=0;
  for(int i=1;i<argc;++i){
    if(!std::strcmp(argv[i],"--M") && i+1<argc) M=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--N") && i+1<argc) N=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--K") && i+1<argc) K=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--it")&& i+1<argc) it=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--use_bias")&& i+1<argc) use_bias=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--relu")&& i+1<argc) relu=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--fused")&& i+1<argc) use_fused=std::atoi(argv[++i]);
  }

  // Setup test data
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f,1.f);

  std::vector<float> Bf((size_t)K*N); 
  for(auto&x:Bf) x=dist(rng);
  
  const int G=64;
  std::vector<uint8_t> Bp; 
  std::vector<uint16_t> Sc;
  
  // Pack B matrix
  std::vector<float> col(K);
  for(int n=0;n<N;++n){ 
    for(int k=0;k<K;++k) col[k]=Bf[(size_t)k*N+n];
    auto pk=q4edge_pack_rowmajor_f32(col.data(),1,K,G);
    Sc.insert(Sc.end(), pk.scales.begin(), pk.scales.end());
    Bp.insert(Bp.end(), pk.data.begin(), pk.data.end());
  }

  std::vector<uint16_t> Ah((size_t)M*K);
  for(int i=0;i<M*K;++i) Ah[i]=f2h(dist(rng));
  
  std::vector<float> bias(N); 
  for(auto &b:bias) b=dist(rng)*0.1f;
  
  std::vector<uint16_t> bias_fp16(N);
  for(int i=0;i<N;++i) bias_fp16[i]=f2h(bias[i]);

  // Benchmark unfused
  auto bench_unfused = [&](){
    std::vector<uint16_t> Ch((size_t)M*N);
    qgemm_int4_fp16_tiled_mt(Ah.data(), K, Bp.data(), Sc.data(), 0, Ch.data(), N, M,N,K,G, std::min(N,8), 8);
    double best=1e100;
    for(int i=0;i<it;++i){
      auto t0=std::chrono::high_resolution_clock::now();
      qgemm_int4_fp16_tiled_mt(Ah.data(), K, Bp.data(), Sc.data(), 0, Ch.data(), N, M,N,K,G, std::min(N,8), 8);
      if (use_bias){
        for(int m=0;m<M;++m){
          for(int n=0;n<N;++n){
            float v=h2f(Ch[(size_t)m*N+n]) + bias[n];
            if (relu && v<0) v=0;
            Ch[(size_t)m*N+n]=f2h(v);
          }
        }
      }
      best = std::min(best, ms_since(t0));
    }
    return best;
  };

#ifdef INT4_FUSE_BIAS
  // Benchmark fused
  auto bench_fused = [&](){
    std::vector<uint16_t> Ch((size_t)M*N);
    qgemm_int4_fp16_tiled_mt_fused(Ah.data(), K, Bp.data(), Sc.data(), 0, Ch.data(), N, 
                                   M,N,K,G, std::min(N,8), 8, 
                                   use_bias ? bias_fp16.data() : nullptr,
                                   relu ? 1 : 0);
    double best=1e100;
    for(int i=0;i<it;++i){
      auto t0=std::chrono::high_resolution_clock::now();
      qgemm_int4_fp16_tiled_mt_fused(Ah.data(), K, Bp.data(), Sc.data(), 0, Ch.data(), N, 
                                     M,N,K,G, std::min(N,8), 8,
                                     use_bias ? bias_fp16.data() : nullptr,
                                     relu ? 1 : 0);
      best = std::min(best, ms_since(t0));
    }
    return best;
  };
#endif

  double gflop=(2.0*M*N*K)/1e9;
  
  double unfused_time = bench_unfused();
  std::printf("Unfused : %.3f ms (%.2f GFLOP/s)\n", unfused_time, gflop/(unfused_time/1000.0));
  
#ifdef INT4_FUSE_BIAS
  if (use_fused) {
    double fused_time = bench_fused();
    std::printf("Fused   : %.3f ms (%.2f GFLOP/s)\n", fused_time, gflop/(fused_time/1000.0));
    std::printf("Speedup : %.2fx (saved %.3f ms)\n", unfused_time/fused_time, unfused_time-fused_time);
  } else {
    std::printf("Fused path available but not tested (use --fused 1)\n");
  }
#else
  std::printf("Fused path not compiled (rebuild with -DINT4_FUSE_BIAS=ON)\n");
#endif
  
  return 0;
}