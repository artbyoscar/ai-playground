# EdgeMind - Experimental High-Performance Quantized Kernels

**⚠️ Research Project - Not Production Ready**

## What This Actually Is

EdgeMind is an experimental project exploring high-performance quantized GEMM (matrix multiplication) kernels for CPU inference. We've achieved **125+ GFLOP/s** on AMD Ryzen 7 8840HS laptop CPU for quantized operations.

**Current Status:** We have fast kernels but they're not yet connected to actual model inference. Think of it as having a high-performance engine that's not yet in a car.

## What Works

✅ **High-performance INT4/INT8 GEMM kernels** - Verified 125+ GFLOP/s  
✅ **Ollama integration wrapper** - Basic Python interface to Ollama models  
✅ **Benchmark suite** - Tests for kernel performance  

## What Doesn't Work Yet

❌ **Kernels don't run actual AI models** - They're orphaned from inference  
❌ **No model loading** - Can't load GGUF/SafeTensors/etc.  
❌ **No quantization pipeline** - Can't convert models to use our kernels  
❌ **Large project size** - Currently 14.4 GB (working on cleanup)  

## Real Performance Numbers

### Kernel Benchmarks (Synthetic)
```
Configuration: AMD Ryzen 7 8840HS
Operation: Quantized GEMM (INT8)
Matrix Size: 256×256×2048

Results:
- 16 threads: 125.31 GFLOP/s 
- 8 threads: 99.35 GFLOP/s
- 4 threads: 78.95 GFLOP/s
- Baseline FP32: ~2.1 GFLOP/s

Speedup: 60× over FP32 (for raw GEMM only)
```

**⚠️ Important:** These benchmarks are for isolated matrix operations, NOT actual model inference. Real model performance will be different.

## Installation

### Prerequisites
- x86-64 CPU with AVX2, FMA, F16C support
- Clang 15+ or GCC 11+
- CMake 3.22+
- Python 3.10+
- Ollama (for model inference)

### Build Kernels (Optional - for testing only)
```bash
# Clone repository
git clone https://github.com/artbyoscar/ai-playground.git
cd ai-playground

# Build performance kernels
cd src/kernels/cpu/int4
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run kernel benchmarks
./build/test_qgemm_perf_q8_mt --M 256 --N 256 --K 2048 --threads 16
```

### Use Ollama Integration (What actually works)
```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama
# macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: Download from https://ollama.com

# Pull a model
ollama pull llama3.2:3b

# Run the wrapper
python src/core/edgemind.py --chat
```

## Current Architecture

```
What we have:
┌─────────────────┐
│  Your Python    │
│     Script      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EdgeMind Wrapper│ ← Just calls Ollama
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Ollama      │ ← Does actual inference
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Output    │
└─────────────────┘

Orphaned kernels:
┌─────────────────┐
│  INT4/INT8      │
│    Kernels      │ ← Fast but unused!
│ (125+ GFLOP/s)  │
└─────────────────┘
```

## The Missing Link

To make this project valuable, we need to build:

```python
# What we need (doesn't exist yet):
def run_inference_with_edgemind(model_path, prompt):
    # Load model weights
    weights = load_gguf(model_path)
    
    # Quantize using our methods
    quantized = quantize_for_edgemind(weights)
    
    # Run through OUR kernels (not Ollama's)
    output = edgemind_kernels.forward(quantized, tokenize(prompt))
    
    return detokenize(output)
```

## Project Structure
```
ai-playground/ (14.4 GB - needs cleanup!)
├── src/
│   ├── kernels/cpu/int4/  # High-performance kernels (working)
│   │   ├── qgemm_int4.cpp  # Core GEMM implementation
│   │   └── tests/          # Performance benchmarks
│   └── core/
│       └── edgemind.py     # Ollama wrapper (not using our kernels)
├── models/                 # Likely contains downloaded models (GB+)
├── build/                  # Build artifacts (should be gitignored)
└── [other files]          # Various experiments
```

## Realistic Roadmap

### Phase 1: Prove Value (Current Priority)
- [ ] Connect kernels to actual model inference
- [ ] Support ONE model end-to-end (suggest Phi-3)
- [ ] Demonstrate real speedup vs. llama.cpp/Ollama
- [ ] Clean up 14.4 GB bloat

### Phase 2: Make Usable (If Phase 1 succeeds)
- [ ] Model loader for GGUF format
- [ ] Quantization pipeline
- [ ] Simple Python API
- [ ] Reduce package to <100 MB

### Phase 3: Consider Open Source (If Phase 2 succeeds)
- [ ] Document thoroughly
- [ ] Create benchmarks vs. competition
- [ ] Build community

## Known Issues

1. **Kernels not connected to inference** - Critical gap
2. **14.4 GB project size** - Needs major cleanup
3. **No model format support** - Can't load models directly
4. **No quantization pipeline** - Can't prepare models for kernels
5. **Windows build issues** - Some tests fail on Windows

## Benchmarking

### Test Kernel Performance (Synthetic)
```bash
cd src/kernels/cpu/int4
./build/test_qgemm_perf_q8_mt --M 256 --N 256 --K 2048 --threads 16
```

### Test Ollama Wrapper (Real inference, but not our kernels)
```bash
python src/core/edgemind.py --benchmark
```

## Contributing

This is an experimental project. Main areas needing work:

1. **Model Loader** - Load GGUF/SafeTensors and convert to our format
2. **Inference Engine** - Connect kernels to actual model forward pass
3. **Quantization** - Convert FP16/FP32 models to INT4/INT8
4. **Size Reduction** - Clean up the 14.4 GB mess

## FAQs

**Q: Can this run ChatGPT locally?**  
A: No. Currently it just wraps Ollama. The fast kernels aren't connected to model inference yet.

**Q: Is 125 GFLOP/s good?**  
A: For synthetic benchmarks, yes. But it doesn't translate to model inference speed yet.

**Q: Why is it 14.4 GB?**  
A: Poor packaging. Likely includes models, build artifacts, and unnecessary files. Core code should be <10 MB.

**Q: When will it be ready?**  
A: Unknown. Connecting kernels to inference is non-trivial. This is research, not a product.

**Q: Should I use this in production?**  
A: Absolutely not. Use Ollama, llama.cpp, or vLLM instead.

## Technical Details

### Kernel Optimizations
- AVX2 SIMD instructions (256-bit vectors)
- F16C for hardware FP16 conversion
- Cache-optimized tiling
- Multi-threaded with OpenMP

### Quantization Methods
- INT8: Symmetric, -127 to 127 range
- INT4: 4-bit weights with FP16 scales
- Group size: 64 (for accuracy)

### Tested Hardware
- AMD Ryzen 7 8840HS (primary development)
- Intel i7-12700K (limited testing)

## Honest Status

**What we claimed:** "Platform for edge AI with multi-API support"  
**What we have:** Fast matrix multiplication that doesn't run AI models yet

**Real value:** The kernels show promise, but until they run actual models, this is just research code.

**Recommendation:** If you need local AI today, use:
- [Ollama](https://ollama.com) - What we currently wrap
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Mature, proven
- [vLLM](https://github.com/vllm-project/vllm) - For GPUs

## License

MIT (for the kernel code that actually exists)

## Author

**Oscar Nuñez**  
Communications & Outreach Specialist, Visual Designer  
Villa Comunitaria - King County, WA  

---

**Note:** This is experimental research code exploring CPU optimization for AI inference. It's not a complete solution. The performance kernels are real but not yet useful for actual AI workloads.