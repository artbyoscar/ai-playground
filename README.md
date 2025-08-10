🧠 EdgeMind v0.5.0 — From “Working Local AI” to Research-Grade Edge Intelligence
Strong claim: We’ve proven local LLMs are practical on consumer laptops. Now we’re turning EdgeMind into a research-driven platform: custom low-bit kernels, quantization-aware adapters, on-device distillation, a reproducible benchmark harness, and a hardened plugin architecture. This README documents what we’ve done, what we learned, where we’re going, and exact next steps.

✅ What We’ve Shipped (v0.4.x → v0.5.0 foundation)
Local-first stack (Ollama) with tested models (Phi-3 Mini, Llama 3.2 3B, DeepSeek R1 7B/14B)

Safety system with supervised guardrails

Smart routing across small models by task intent

Conversation memory with context trimming (~10 exchanges)

Practical apps: assistant, code reviewer, web UI

Benchmark script for repeatable local timing

Quantization utilities (INT4 prototype) and BitNet 1-bit layer scaffold (R&D)

New in v0.5.0 workstream (Aug 10, 2025)
INT4 CPU QGEMM (AVX2+F16C) with group=64 and per-group fp16 scales

Kernels: single-thread, multithread, and tiled MT (reuse A across N-tiles)

Correctness harness vs FP32 ref with threshold = 8e-2 → PASS on multiple shapes

Python pack/check tools (colwise K-blocks, fp16 scales) → rel_err ≈ 0.057

Perf harness: prints GFLOP/s and speedups vs FP32

DML subtree scaffold with a smoke test (placeholder target)

CMake + CTest wiring (Windows/Clang/Ninja path verified)

🔍 What We Learned
Performance & UX

Latency is often memory-bound. INT4 helps, but layout + cache reuse + prefetch matter more.

Routing > “one big model.” Smaller, specialized models win for most tasks.

Caching is king. Tiered cache makes repeats drop from seconds → milliseconds.

Quantization & accuracy

INT4 per-group works well for chat/coding; more aggressive bits (INT2/1) will want QAT/adapter help.

Safety & eval

Prompt-only filters are brittle; tool allow-lists reduce false blocks.

Benchmarks must combine quality + latency. Token/sec alone is misleading.

Developer experience

CLI and Web UI parity reduces friction.

Clear plugin boundaries keep the codebase maintainable.

🧪 Latest R&D Results (Aug 10, 2025)
Correctness (C++ harness, --threshold 8e-2): PASS
Representative shape set:

M=32,N=32,K=256 → rel≈0.0666

M=64,N=64,K=512 → rel≈0.0703

M=64,N=64,K=4096 → rel≈0.0703

M=48,N=48,K=1000 → rel≈0.0707

Packing sanity (Python): rel_err ≈ 5.7e-2

Perf vs FP32 baseline (one recent run, M=256,N=256,K=2048, it=5):

FP32 ref: 214.0 ms (1.25 GFLOP/s)

st: ~20.0 ms (13.4 GFLOP/s) → 10.7×

mt: ~5.34 ms (50.2 GFLOP/s) → 40.1×

tmt: ~4.50 ms (59.7 GFLOP/s) → 47.6×

These are early numbers. We haven’t added M-tiling, bias/activation fusion, or deeper cache-aware K-blocking yet.

🏗 Architecture (v0.5 direction)
Core: Router → Model Runner (Ollama / llama.cpp) → Tool Layer (plugins with scopes)

Optimization: Quantizer (PTQ/QAT), Kernel backends (CPU/DirectML), Cache (L1/L2 + KV)

Memory: Rolling context, eviction policy, per-task compression

Safety: Policy + tool allowlist, per-plugin guardrails, audit log

Offline-first by default; online tools are opt-in

⚙️ Quick Start (Working Today)
bash
Copy code
# Clone & setup
git clone https://github.com/artbyoscar/ai-playground.git
cd ai-playground
python -m venv ai-env
# Windows
.\ai-env\Scripts\Activate.ps1
# macOS/Linux
source ai-env/bin/activate
pip install -r requirements.txt
pip install ollama
bash
Copy code
# Install models (pick a few)
ollama pull phi3:mini
ollama pull llama3.2:3b
ollama pull deepseek-r1:7b-qwen-distill-q4_k_m
bash
Copy code
# Run chat
python src/core/edgemind.py --chat
# Demos / benchmarks
python demo.py
python src/core/edgemind.py --benchmark
🔧 Building the CPU INT4 Kernel (Windows / LLVM + Ninja)
powershell
Copy code
# From repo root
cd src/kernels/cpu/int4
Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release -v
# sanity
ctest --test-dir .\build -R correctness -V
.\build\test_qgemm_perf_vs_baseline.exe --M 256 --N 256 --K 2048 --it 5
Notes

Requires AVX2 + F16C. Uses per-group fp16 scales and 64-element groups.

fp16 conversion uses a minimal helper; we’ll swap in precise intrinsics or a lib wrapper later.

See tools/quant/ for packers and error checks.

📊 Current Local Performance (Laptop, 16 GB) — Anecdotal
Model	Size	tok/s	Notes
phi3:mini	2.2G	~5.8	fastest replies
llama3.2:3b	2.0G	~7.9	balanced chat
deepseek-r1:7b	4.7G	~4.7	coding tasks
deepseek-r1:14b	9.0G	2–3	complex reasoning

Kernel targets (near term):

2× throughput on 3B–7B class with INT4 kernels

≤0.5% quality delta via QAT adapters

Student models that beat base 3B/7B on user tasks

🧪 Benchmark Suite (what we’ll report)
Latency: end-to-end, first token, throughput

Quality proxies: shortform QA, code solve rate (toy), retrieval hit rate

Safety: false-block / false-allow by category

Memory: peak RAM, model load time, cache effectiveness

Reproducibility: hardware profile + fixed seeds

🔐 Safety Approach
Model-level policies: refusal templates, disallowed categories

Tool scopes: allowlists, rate limits, “dry-run” for risky ops

Audit logs: tool calls, sources, decisions; user-clearable

Offline-first by default; online tools are opt-in

🧰 Developer Guide
Plugins: manifest (name, scope, params, I/O), tests required

Kernels: contrib backends under src/kernels/ (CPU/DirectML) + clear benchmarks

Quantization: tools/quant/ for PTQ + QAT flows, GGUF export scripts

Distillation: distill/ background trainer, reward heuristics, pack exporter

🧭 Roadmap (Q3–Q4 2025)
August–September (v0.5.0-alpha)

INT4 kernel prototype (CPU), PTQ → GGUF pipeline (in progress)

QAT/LoRA adapters at 4-bit on a 3B model

Benchmark harness v1 (CLI + HTML)

October (v0.5.0)

DirectML path for AMD iGPU (INT4)

On-device distillation (teacher sampling + idle training)

Plugin SDK with guardrails, 3 reference plugins (search, RAG, code exec)

November–December (v0.5.1)

INT2 experimental kernels + FP8 activations

Distillation “packs” export/import

Benchmark site with community submissions

🎯 Next Steps (Actionable)
Kernel

Tighten correctness threshold to 6e-2 after a few more shape sweeps

Add M-blocked variant to the test driver; expose --m_tile

JSON output mode for perf harness (store in CI artifacts)

Bias/activation fusion path; deeper prefetch + cache-aware K-blocking

Quantization

Add packers for per-channel/per-row scales

GGUF export path; llama.cpp patch notes

CI

Windows workflow (build + correctness + python sanity); add perf JSON as artifact

Smoke test DML subtree, then swap in real DirectML kernels

Docs

Small diagrams for packing layout and kernel tiling

📝 Changelog Highlights
v0.4.x: Local stack stabilized; routing, safety, memory, demos; INT4 PTQ & BitNet scaffolds.

v0.5.0 (in progress): INT4 kernels (CPU/DirectML), QAT adapters, on-device distillation, benchmark harness v1, plugin SDK.

📄 License
MIT — free to use, modify, and distribute.

🧩 The Bottom Line
We turned “local LLMs on laptops” into a real product.
Now we’re executing a research-grade plan: custom low-bit kernels, QAT adapters, on-device distillation, reproducible benchmarks, and a safe plugin system.
Start here: python src/core/edgemind.py --chat
Repo: github.com/artbyoscar/ai-playground