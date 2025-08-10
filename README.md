🧠 EdgeMind v0.5.0 — From “Working Local AI” to Research-Grade Edge Intelligence
Strong claim: We’ve proven local LLMs are practical on consumer laptops. Now we’re turning EdgeMind into a novel, research-driven platform: custom low-bit kernels, quantization-aware adapters, on-device distillation, a reproducible benchmark harness, and a hardened plugin architecture. This README documents what we’ve done, what we learned, where we’re going, and exact next steps.

✅ What We’ve Shipped (v0.4.x → v0.5.0 foundation)
Local-first stack (Ollama) with tested models (Phi-3 Mini, Llama 3.2 3B, DeepSeek R1 7B/14B)

Safety system with supervised guardrails

Smart routing across small models by task intent

Conversation memory with context trimming (~10 exchanges)

Practical apps: assistant, code reviewer, web UI

Benchmark script for repeatable local timing

Quantization utilities (INT4 prototype) and BitNet 1-bit layer scaffold (R&D)

“BEAST” demo wiring quantization + BitNet + routing + cache into one run

New in v0.5.0 workstream (Aug 10, 2025):

INT4 CPU QGEMM prototype (AVX2+F16C) with group-size 64 and per-group fp16 scales

Multithreaded + tiled kernels (column tiling to reuse A; thread scaling verified)

Example wins:

On a Lenovo Yoga (16 GB), Phi-3 Mini and Llama 3.2 3B are usable for daily tasks.

Safety refusals fire on risky prompts; normal dev tasks pass.

Routing to DeepSeek 7B improves code results without wrecking latency.

🔍 What We Learned
Performance & UX

Latency is often memory-bound. INT4 helps, but kernel layout and prefetch matter more.

Routing > one big model for laptops. Specialized small models win for most tasks.

Caching is king. Tiered cache makes repeats drop from seconds to milliseconds.

Quantization & accuracy

INT4 per-channel is solid for chat/coding. INT2/1-bit will need QAT/adapter help to keep quality.

Safety & eval

Prompt-only filters are brittle. Tool-level allowlists reduce false blocks.

Benchmarks must combine quality + latency. Token/sec alone is misleading.

Developer experience

CLI + Web UI parity reduces friction.

Plugin boundaries keep the codebase maintainable.

🚀 Where We’re Going (Novelty & Moat)
1) Custom Low-Bit Kernels (AMD iGPU/CPU)
Goal: INT4/INT2 + 1-bit matmul kernels targeting DirectML and LLVM; optional FP8 activations.

Why it’s novel: Co-design quantization with hardware-aware kernels for consumer AMD APUs.

Deliverables: QGEMM kernels (INT4 weights, FP16/FP8 activations), packing/layout (per-channel scales, 64-group blocks), llama.cpp patch + GGUF path.

Success: ≥ 2× throughput vs. current CPU path on 7B models; ≤1% quality delta on small evals.

2) Quantization-Aware Adapters (QAT/LoRA at 2–4 bit)
Goal: Keep accuracy at low bits via LoRA-style adapters with QAT, auto-export to GGUF.

Deliverables: QAT pipeline (INT4/INT2), adapter export/merge, one-click tuning script.

Success: ≤0.5% drop vs. FP16 on held-out tasks; drop-in with Ollama.

3) On-Device Distillation (Teacher → Student, offline)
Goal: Use cached Q&A + optional RAG to self-distill into 3–7B students while idle.

Deliverables: Reward-guided sampling, idle mini-epochs (power-aware), “personalization packs”.

Success: Local student beats base 3B/7B on user tasks at same or lower latency.

4) Reproducible Benchmark Suite
Goal: Public harness for latency, throughput, quality, memory, safety on laptops.

Deliverables: Deterministic prompts, metrics (tok/s, E2E latency, accuracy proxies, refusal stats), JSON + HTML report.

Success: Reproducible within ±5% on the same hardware class.

5) Hardened Plugin Architecture
Goal: Route tools (search, code exec, RAG, vision, OS control) per prompt signature with guardrails.

Deliverables: @tool manifest (scopes, rate limits), router with decision logs, unit + fuzz tests.

Success: Safe-by-default, auditable, easy to extend.

🧪 Latest R&D Results (Aug 10, 2025)
CPU INT4 QGEMM (AVX2+F16C, group=64, per-group fp16 scales)

Single-thread microbench: M=64, N=64, K=4096 → ~2.40 ms (~14 GFLOP/s)

Multithread sweep (same shape): 8 threads ~38 GFLOP/s, saturates ~8–16 threads

Tiled kernel sweeps show best bands around T=4–8, tile NC=4–8

These numbers are a floor — we haven’t added M-tiling, bias/activation fusion, or cache-aware K-blocking yet.

🏗 Architecture (v0.5 direction)
Core: Router → Model Runner (Ollama / llama.cpp) → Tool Layer (plugins with scopes)

Optimization: Quantizer (PTQ/QAT), Kernel backends (CPU/DirectML), Cache (L1/L2 + KV)

Memory: Rolling context, eviction policy, per-task compression

Safety: Policy + tool allowlist, per-plugin guardrails, audit log

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
.\build\test_qgemm.exe
.\build\test_qgemm_perf.exe
.\build\test_qgemm_perf_tiled.exe
Notes

Requires AVX2 + F16C. Uses per-group fp16 scales and 64-element groups.

fp16 conversion is a minimal helper for now; will replace with precise conversion libs.

See tools/quant/ for packers and error checks.

📊 Current Local Performance (Laptop, 16 GB)
Model	Size	tok/s	Notes
phi3:mini	2.2GB	~5.8	fastest replies
llama3.2:3b	2.0GB	~7.9	balanced chat
deepseek-r1:7b	4.7GB	~4.7	coding tasks
deepseek-r1:14b	9.0GB	2–3	complex reasoning

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

INT4 kernel prototype (CPU), PTQ → GGUF pipeline

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
This week

Ship INT4 CPU QGEMM prototype + microbenchmarks (done)

Add correctness harness + error thresholds (next)

QAT adapter training for a 3B chat model at 4-bit

Benchmark harness v1 (CLI + HTML report)

Next 30 days

DirectML INT4 path on AMD iGPU

Idle-time distillation loop with reward-guided sampling

Plugin SDK with 3 reference tools + guardrails

Quarter goal (Q4)

Public results page with reproducible runs

Student model beats base 3B on user tasks at same latency

2× throughput on 7B class with kernels

📝 Changelog Highlights
v0.4.x: Local stack stabilized; routing, safety, memory, demos; INT4 PTQ & BitNet scaffolds.

v0.5.0 (in progress): INT4 kernels (CPU/DirectML), QAT adapters, on-device distillation, benchmark harness v1, plugin SDK.

📄 License
MIT — free to use, modify, and distribute.

🧩 The Bottom Line
We turned “local LLMs on laptops” into a real product.

Now we’re executing a research-grade plan: custom low-bit kernels, QAT adapters, on-device distillation, reproducible benchmarks, and a safe plugin system.

That’s how EdgeMind stays useful today and becomes novel tomorrow.

Start here: python src/core/edgemind.py --chat
Repo: github.com/artbyoscar/ai-playground