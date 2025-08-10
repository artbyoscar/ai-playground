
# 🧠 EdgeMind v0.5.0 — From “Working Local AI” to Research-Grade Edge Intelligence

**Strong claim:** We’ve proven local LLMs are practical on consumer laptops. Now we’re turning EdgeMind into a research-driven platform: custom low-bit kernels, quantization-aware adapters, on-device distillation, a reproducible benchmark harness, and a hardened plugin architecture.

This README documents what we’ve done, what we learned, where we’re going, and exact next steps.

## ✅ What We’ve Shipped (v0.4.x → v0.5.0 foundation)

* Local-first stack (Ollama) with tested models (Phi-3 Mini, Llama 3.2 3B, DeepSeek R1 7B/14B)
* Safety system with supervised guardrails
* Smart routing across small models by task intent
* Conversation memory with context trimming (\~10 exchanges)
* Practical apps: assistant, code reviewer, web UI
* Benchmark script for repeatable local timing
* Quantization utilities (INT4 prototype) and BitNet 1-bit layer scaffold (R\&D)

## New in v0.5.0 workstream (Aug 10, 2025)

* **INT4 CPU QGEMM (AVX2+F16C)** with `group=64` and per-group FP16 scales
* Kernels: **single-thread (st), multithread (mt), tiled MT (tmt)** + **M-blocked tiled prototype**
* **Correctness harness** vs FP32 ref — **threshold = `7.2e-2` → PASS** on multiple shapes
* **Python pack/check tools** (col-wise K-blocks, FP16 scales) → rel\_err ≈ **5.7e-2**
* **Packed loader + perf harness** to run kernels directly on on-disk packs
* **DML subtree scaffold** with a smoke test (placeholder target)
* **CMake + CTest wiring** (Windows/Clang/Ninja path verified) with PATH guidance

---

## 🔍 What We Learned

**Performance & UX**
Latency is often memory-bound. INT4 helps, but **layout + cache reuse + tiling + prefetch** matter more.
Routing > one big model; small, specialized models win most tasks.
Caching is king—tiered cache takes repeats from seconds → milliseconds.

**Quantization & accuracy**
INT4 per-group works well for chat/coding; more aggressive bits (INT2/1) likely need **QAT/adapters**.

**Safety & eval**
Prompt-only filters are brittle; **tool allow-lists** reduce false blocks.
Benchmarks must combine **quality + latency**—tok/s alone is misleading.

**Developer experience**
CLI/Web UI parity reduces friction. Clear plugin boundaries keep the codebase maintainable.

---

## 🧪 Latest R\&D Results (Aug 10, 2025)

**Correctness (C++ harness, `--threshold 7.2e-2`): PASS**
Representative shapes (st/mt/tmt are identical here by design):

* M=32, N=32, K=256 → rel ≈ **0.0666**
* M=64, N=64, K=512 → rel ≈ **0.0703**
* M=64, N=64, K=4096 → rel ≈ **0.0703**
* M=48, N=48, K=1000 → rel ≈ **0.0707**

**Packing sanity (Python):** rel\_err ≈ **5.71e-2** (K=2048, N=256, group=64)

**Perf vs FP32 baseline** (one recent run, **M=256, N=256, K=2048, it=5**):

* FP32 ref: **\~235 ms** (≈ **1.14 GFLOP/s**)
* **st**: \~**21.0 ms** (≈ **12.8 GFLOP/s**) → **11.2×**
* **mt**: \~**4.68 ms** (≈ **57.3 GFLOP/s**) → **50.2×**
* **tmt**: \~**4.42 ms** (≈ **60.7 GFLOP/s**) → **53.2×**
* **tmt+m (prototype)**: \~**4.32 ms** (≈ **62.2 GFLOP/s**) → **54.5×**

**Perf vs packed B (loader path)**

* **st**: \~**18.8 ms** (≈ **14.3 GFLOP/s**)
* **mt**: \~**4.03 ms** (≈ **66.7 GFLOP/s**)
* **tmt**: \~**3.66 ms** (≈ **73.4 GFLOP/s**)

> Perf varies with power/thermals (±10–20%). Re-run locally for apples-to-apples.

---

## 🏗 Architecture (v0.5 direction)

**Core:** Router → Model Runner (Ollama / llama.cpp) → Tool Layer (plugins with scopes)
**Optimization:** Quantizer (PTQ/QAT), Kernel backends (CPU/DirectML), Cache (L1/L2 + KV)
**Memory:** Rolling context, eviction policy, per-task compression
**Safety:** Policy + tool allowlist, per-plugin guardrails, audit log
**Offline-first** by default; online tools are opt-in

---

## ⚙️ Quick Start (Working Today)

```bash
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
```

```bash
# Install models (pick a few)
ollama pull phi3:mini
ollama pull llama3.2:3b
ollama pull deepseek-r1:7b-qwen-distill-q4_k_m
```

```bash
# Run chat
python src/core/edgemind.py --chat
# Demos / benchmarks
python demo.py
python src/core/edgemind.py --benchmark
```

---

## 🔧 Building the CPU INT4 Kernel (Windows / LLVM + Ninja)

**Prereqs:** AVX2 + F16C CPU, LLVM/Clang (tested 20.1.8), CMake ≥ 3.22, Ninja, Python 3.13 + NumPy.
**PATH tip (PowerShell):**

```powershell
$env:Path = "C:\Program Files\LLVM\bin;C:\Program Files\CMake\bin;$env:AppData\Python\Python313\Scripts;$env:Path"
```

From `src/kernels/cpu/int4`:

```powershell
# Generate a random test matrix + quantize
python .\B_gen.py
python ..\..\..\..\tools\quant\quantize_q4_edge.py --weights .\B.npy --out .\pack\q4edge --group 64

# Configure + build
Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release

# Correctness (current threshold = 7.2e-2)
ctest --test-dir .\build -R correctness -V

# Perf vs FP32 baseline
.\build\test_qgemm_perf_vs_baseline.exe --M 256 --N 256 --K 2048 --it 5

# Perf vs packed (loader path)
.\build\test_perf_load_packed.exe --packed .\pack\q4edge --M 256 --it 5
```
### Third-party SDKs (Windows)
We don’t commit vendor SDK binaries. Hydrate them on demand:


**Notes**

* Uses per-group FP16 scales and 64-element groups (column-wise K blocks; 2 nibbles/byte).
* FP16 conversion has a precise scalar helper; F16C fast path used when available.
* See `tools/quant/` for packers and error checks.
* PowerShell doesn’t support `python - <<'PY'` heredocs—save to a `.py` file instead.

---

## 📊 Current Local Performance (Laptop, 16 GB) — Anecdotal

| Model           | Size | tok/s | Notes             |
| --------------- | ---- | ----: | ----------------- |
| phi3\:mini      | 2.2G | \~5.8 | fastest replies   |
| llama3.2:3b     | 2.0G | \~7.9 | balanced chat     |
| deepseek-r1:7b  | 4.7G | \~4.7 | coding tasks      |
| deepseek-r1:14b | 9.0G |   2–3 | complex reasoning |

**Kernel targets (near term):**

* ≥2× throughput on 3B–7B class with INT4 kernels
* ≤0.5% quality delta via QAT adapters
* Student models that beat base 3B/7B on user tasks

---

## 🧪 Benchmark Suite (what we’ll report)

* Latency: end-to-end, first token, throughput
* Quality proxies: shortform QA, code solve rate (toy), retrieval hit rate
* Safety: false-block / false-allow by category
* Memory: peak RAM, model load time, cache effectiveness
* Reproducibility: hardware profile + fixed seeds

---

## 🔐 Safety Approach

* Model-level policies: refusal templates, disallowed categories
* Tool scopes: allowlists, rate limits, “dry-run” for risky ops
* Audit logs: tool calls, sources, decisions; user-clearable
* Offline-first by default; online tools are opt-in

---

## 🧰 Developer Guide

* Plugins: manifest (name, scope, params, I/O), tests required
* Kernels: contrib backends under `src/kernels/` (CPU/DirectML) + clear benchmarks
* Quantization: `tools/quant/` for PTQ + QAT flows, GGUF export scripts
* Distillation: `distill/` background trainer, reward heuristics, pack exporter

---

## 🧭 Roadmap (Q3–Q4 2025)

**August–September (v0.5.0-alpha)**

* INT4 kernel prototype (CPU) + PTQ → GGUF pipeline (in progress)
* QAT/LoRA adapters at 4-bit on a 3B model
* Benchmark harness v1 (CLI + HTML)

**October (v0.5.0)**

* DirectML path for AMD iGPU (INT4)
* On-device distillation (teacher sampling + idle training)
* Plugin SDK with guardrails, 3 reference plugins (search, RAG, code exec)

**November–December (v0.5.1)**

* INT2 experimental kernels + FP8 activations
* Distillation “packs” export/import
* Benchmark site with community submissions

---

## 🎯 Next Steps (Actionable)

**Kernel**

* Tighten correctness to **7e-2** (stretch: **6e-2** with rounding tweaks/QAT)
* Promote **M-blocked tiled** kernel to first-class CLI with `--threads`, `--nc-tile`, `--m-tile`
* **Epilogue fusion** (bias + ReLU/SILU/GELU)
* Prefetch polish and deeper cache-aware K-blocking

**Quantization**

* Add packers for **per-channel/per-row scales**
* GGUF export path + llama.cpp patch notes

**CI**

* Windows workflow: build + correctness + Python pack sanity; publish **perf CSV/JSON** as artifact
* DML smoke → swap in a real DirectML kernel consuming the same pack

**Docs**

* Small diagrams for **packing layout** and **kernel tiling**
* Troubleshooting (PATH, Ninja, PowerShell heredoc)

---

## 📝 Changelog Highlights

**v0.4.x:** Local stack stabilized; routing, safety, memory, demos; INT4 PTQ & BitNet scaffolds.
**v0.5.0 (in progress):** INT4 kernels (CPU/DirectML), QAT adapters, on-device distillation, benchmark harness v1, plugin SDK.

---

## 📄 License

MIT — free to use, modify, and distribute.

---

**Start here:** `python src/core/edgemind.py --chat`
**Repo:** `github.com/artbyoscar/ai-playground`

---

### Suggested commit message

```
docs(README): update v0.5.0 workstream with INT4 CPU results, loader path, and build/test instructions

- Correctness now passing at 7.2e-2 across multiple shapes
- Added packed loader + perf harness; included commands and PATH tips
- Documented Python pack/check tools (rel_err ~5.7e-2)
- Included latest perf (st/mt/tmt and M-blocked prototype; packed path)
- Added near-term next steps (threshold tighten, epilogue fusion, CI, DML)
```
