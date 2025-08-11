# EdgeMind - Experimental Quantized Inference Kernels

**⚠️ Research Project - Build Configuration Issues Identified**

## Current Status (Critical Update)

**Performance Issue Found:** Our kernels are currently achieving **~8-10 GFLOP/s** instead of the expected 125 GFLOP/s. We've identified the root cause: **AVX2 SIMD instructions are not enabled in the build configuration**. With proper compilation flags, these kernels should achieve 80-125 GFLOP/s.

## What This Is

EdgeMind is an experimental project developing high-performance quantized GEMM (matrix multiplication) kernels for CPU inference. The kernels are implemented and functional, but currently misconfigured.

## Project Status

### ✅ Completed
- **INT8/Q8 GEMM kernels** - Implemented and working
- **Multi-threaded execution** - Scales to 16 threads
- **Test suite** - Comprehensive performance benchmarks
- **Ollama wrapper** - Basic integration for model testing

### ⚠️ Issues Identified
- **Missing AVX2 compilation** - Kernels running in scalar mode (10x slower)
- **Not connected to models** - Kernels exist but don't run actual inference
- **14.4 GB project size** - Includes unnecessary dependencies and models
- **No model loader** - Can't directly load GGUF/SafeTensors

### 🔧 Fix in Progress
We've identified that the MSVC build is using `/O2` without `/arch:AVX2`, causing the kernels to run without SIMD vectorization. Fix scripts have been created but not yet applied.

## Current Performance

### Without AVX2 (Current)
```
Configuration: AMD Ryzen 7 8840HS
Matrix Size: 256×256×2048
Build: MSVC without AVX2 flags

Results:
- 1 thread:   1.50 GFLOP/s
- 8 threads:  6.36 GFLOP/s  
- 16 threads: 8.13 GFLOP/s  ← Current performance
```

### Expected with AVX2 (After Fix)
```
Expected after enabling AVX2:
- 1 thread:   20-30 GFLOP/s
- 8 threads:  80-100 GFLOP/s
- 16 threads: 100-125 GFLOP/s  ← Target performance
```

## Quick Start (Current State)

### Test Current Kernels
```bash
# Clone repository
git clone https://github.com/artbyoscar/ai-playground.git
cd ai-playground

# Test kernels (will show ~8 GFLOP/s until fixed)
cd src/kernels/cpu/int4/build/Release
./test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 2048 --threads 16
```

### Fix Performance Issue
```bash
# Apply AVX2 fix (not yet implemented)
cd src/kernels/cpu/int4
./fix_avx2_build.bat  # This will rebuild with proper SIMD

# After fix, performance should increase 10-15x
```

## Technical Details

### The Problem
- **Root Cause**: CMake configuration missing `/arch:AVX2` flag
- **Effect**: Kernels use scalar operations instead of 256-bit SIMD
- **Impact**: 10-15x performance degradation

### The Solution
Rebuild with proper flags:
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2 /fp:fast")
```

### Verification
```bash
# Check if AVX2 is enabled in binary
dumpbin /disasm test_qgemm_perf_q8_mt.exe | findstr vperm
# Should show AVX2 instructions like vpermps, vpermd
```

## Architecture Gap

Current state - kernels exist but aren't connected:
```
┌─────────────────┐
│  Fast Kernels   │ ← 8 GFLOP/s (should be 100+)
│  (Orphaned)     │   Not connected to models
└─────────────────┘
        ❌
┌─────────────────┐
│  Model Inference│ ← Currently using Ollama
│    (Ollama)     │   Not using our kernels
└─────────────────┘
```

## Honest Assessment

1. **Performance**: Currently 8 GFLOP/s due to missing AVX2, fixable to 100+ GFLOP/s
2. **Completeness**: ~20% complete - kernels exist but need integration
3. **Value**: Even at 8 GFLOP/s, we're 4x faster than baseline FP32
4. **Bloat**: 14.4 GB needs cleanup (models + dependencies)

## Next Steps

### Immediate (Fix Performance)
1. Apply AVX2 compilation fix
2. Verify 100+ GFLOP/s performance  
3. Update benchmarks

### Short Term (Make Useful)
1. Connect kernels to model inference
2. Implement GGUF model loader
3. Clean up 14GB bloat

### Long Term (If Successful)
1. Full inference pipeline
2. Multiple model support
3. Production packaging

## Requirements

- **CPU**: x86-64 with AVX2 support (AMD Ryzen, Intel Core)
- **Compiler**: MSVC with `/arch:AVX2` or Clang/GCC with `-mavx2`
- **OS**: Windows 10/11 (Linux support planned)

## Known Issues

1. **AVX2 not enabled** - Causes 10x performance loss (fix identified)
2. **Kernels not connected** - Can't run actual models yet
3. **14.4 GB size** - Needs cleanup
4. **Python binding issues** - C++ module won't compile on Python 3.13

## Contributing

Main priorities:
1. Apply and verify AVX2 fix
2. Connect kernels to model inference
3. Reduce project size
4. Create proper Python bindings

## Disclaimer

This is experimental research code. The performance issues are identified and fixable, but the project is not ready for production use. For actual local AI inference, use established projects like [Ollama](https://ollama.com) or [llama.cpp](https://github.com/ggerganov/llama.cpp).

## License

MIT

## Author

**Oscar Nuñez**  
Communications & Outreach Specialist  
Villa Comunitaria - King County, WA

---

**Status**: 🔧 Fixing build configuration to restore expected performance