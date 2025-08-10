# 🧠 EdgeMind v0.5.0 — From “Working Local AI” to Research-Grade Edge Intelligence

**Strong claim:** We’ve proven local LLMs are practical on consumer laptops. Now we’re turning EdgeMind into a **novel, research-driven platform**: custom low-bit kernels, quantization-aware adapters, on-device distillation, a reproducible benchmark harness, and a hardened plugin architecture. This README documents **what we’ve done**, **what we learned**, **where we’re going**, and **exact next steps**.

---

## ✅ What We’ve Shipped (v0.4.x → v0.5.0 foundation)

* **Local-first stack (Ollama)** with tested models (Phi-3 Mini, Llama 3.2 3B, DeepSeek R1 7B/14B)
* **Safety system** with supervised guardrails
* **Smart routing** across small models by task intent
* **Conversation memory** with context trimming (≈10 exchanges)
* **Practical apps**: assistant, code reviewer, web UI
* **Benchmark script** for repeatable local timing
* **Quantization utilities** (INT4 prototype) and BitNet 1-bit layer scaffold (R\&D)
* **“BEAST” demo** that wires quantization + BitNet + routing + cache into one run

> Example wins:
>
> * On a **Lenovo Yoga / 16GB RAM**, Phi-3 Mini and Llama 3.2 3B run at usable speeds for daily work.
> * Safety refusals fire on obviously risky prompts while allowing normal dev tasks.
> * Routing to DeepSeek 7B boosts code-related accuracy without collapsing latency.

---

## 🔍 What We Learned

**Performance & UX**

* **Latency ceiling is memory-bound.** INT4 quantization helps, but kernels and memory layout dominate.
* **Routing > one big model.** Small, specialized models beat a bigger generic model for most laptop tasks.
* **Caching is king.** Tiered cache turns repeats from seconds → milliseconds; needs dedupe + invalidation.

**Quantization & accuracy**

* **INT4 per-channel quant** keeps quality acceptable for chat/coding; INT2/1-bit need adapter/QAT help.
* **1-bit (BitNet-style) layers** are promising, but without specialized kernels the gains are limited on CPU.

**Safety & eval**

* **Guardrails must be measured.** Prompt-only filters are brittle; tool-level allowlists reduce false blocks.
* **Benchmarks matter.** Token/sec alone is misleading; we need a **quality+latency** score that reflects real tasks.

**Developer experience**

* **CLI + Web UI parity** reduces friction.
* **Plugin boundaries** prevent “giant ball of mud” as features grow.

---

## 🚀 Where We’re Going (Novelty & Moat)

### 1) Custom Low-Bit Kernels (AMD iGPU/CPU)

* **Goal:** INT4/INT2 + 1-bit matmul kernels targeting **DirectML** and **LLVM** backends; optional FP8 activations.
* **Why it’s novel:** Co-design quantization with **hardware-aware kernels** for consumer AMD APUs.
* **Deliverables:**

  * QGEMM kernels (INT4 weights, FP16/FP8 activations)
  * Packing/layout (per-channel scales, 64-group blocks)
  * llama.cpp patch + GGUF path for easy adoption
* **Success:** ≥ **2× throughput** vs. current CPU path on 7B models; **≤1%** quality delta on small evals.

### 2) Quantization-Aware Adapters (QAT/LoRA at 2–4 bit)

* **Goal:** Keep accuracy at low bits by training **LoRA-style adapters** with QAT, **auto-export to GGUF**.
* **Deliverables:**

  * QAT pipeline (INT4/INT2) with per-channel scales
  * Adapter export that merges into quantized weights
  * “1-click” QAT tuning script for user datasets
* **Success:** **≤0.5%** drop on held-out tasks vs. FP16 baselines; **drop-in** with Ollama.

### 3) On-Device Distillation (Teacher → Student in the background)

* **Goal:** Use cached Q\&A + optional RAG to **self-distill** into 3–7B students, entirely offline.
* **Deliverables:**

  * Reward-guided sampling (choose best teacher answers)
  * Mini-epochs that run when idle (power-aware)
  * “Personalization packs” the user can export/share
* **Success:** Local student beats base 3B/7B on user’s tasks at same or lower latency.

### 4) Reproducible Benchmark Suite (Consumer-Laptop-Ready)

* **Goal:** Public harness to measure **latency, throughput, quality, memory, and safety** on laptops.
* **Deliverables:**

  * Deterministic prompts for chat, code, RAG, safety
  * Metrics: token/sec, end-to-end latency, accuracy proxies, refusal stats
  * Results JSON + pretty HTML report; CI artifacts
* **Success:** Any contributor can reproduce within **±5%** on the same hardware class.

### 5) Hardened Plugin Architecture (Tools with Guardrails)

* **Goal:** Route search, code exec, RAG, vision, and OS control **per prompt signature** with **permission gates**.
* **Deliverables:**

  * `@tool` manifest (scopes, rate limits, red lines)
  * Router that selects models/tools, logs decisions
  * Unit tests + fuzz tests for each tool
* **Success:** Plugins are safe by default, auditable, and easy to add.

---

## 🧭 Roadmap (Q3–Q4 2025)

**August–September (v0.5.0-alpha)**

* INT4 kernel prototype (CPU first), PTQ → GGUF pipeline
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

## 🏗 Architecture (v0.5 direction)

* **Core:** Router → Model Runner (Ollama / llama.cpp) → Tool Layer (plugins with scopes)
* **Optimization:** Quantizer (PTQ/QAT), Kernel backends (CPU/DirectML), Cache (L1/L2 + KV)
* **Memory:** Rolling context, eviction policy, per-task compression
* **Safety:** Policy + tool allowlist, per-plugin guardrails, audit log

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
# Demo / benchmarks
python demo.py
python src/core/edgemind.py --benchmark
```

---

## 📊 Current Local Performance (Laptop, 16GB RAM)

| Model           |  Size | tok/s | Notes             |
| --------------- | ----: | ----: | ----------------- |
| phi3\:mini      | 2.2GB | \~5.8 | fastest replies   |
| llama3.2:3b     | 2.0GB | \~7.9 | balanced chat     |
| deepseek-r1:7b  | 4.7GB | \~4.7 | coding tasks      |
| deepseek-r1:14b | 9.0GB |   2–3 | complex reasoning |

> Targets with kernels & QAT:
>
> * **2×** throughput on 3B–7B (INT4 kernels)
> * **≤0.5%** quality delta via QAT adapters
> * **Student models** that beat base 3B/7B on user tasks

---

## 🧪 Benchmark Suite (what we’ll report)

* **Latency:** end-to-end, first token, throughput
* **Quality proxies:** shortform QA, code solve rate (toy), retrieval hit rate
* **Safety:** false-block / false-allow counts by category
* **Memory:** peak RAM, model load time, cache effectiveness
* **Reproducibility:** hardware profile + fixed seeds

---

## 🔐 Safety Approach

* **Model-level policies**: refusal templates, disallowed categories
* **Tool scopes**: explicit allowlists, rate limits, “dry-run” mode for risky ops
* **Audit logs**: tool calls, sources, decisions; user-clearable
* **Offline-first** by default; online tools require opt-in

---

## 🧰 Developer Guide

* **Plugins:** simple manifest (name, scope, params, I/O), tests required
* **Kernels:** contrib backends live under `kernels/` (CPU/DirectML), clear benchmarks
* **Quantization:** `quant/` with PTQ + QAT flows, GGUF export scripts
* **Distillation:** `distill/` background trainer, reward heuristics, pack exporter

---

## 🤝 Contributing

**We need help with:**

* INT4/INT2/1-bit kernels (DirectML/LLVM)
* QAT/LoRA adapters and accuracy evals
* On-device distillation heuristics
* Benchmark harness + website
* Plugin examples (search, RAG, vision) and tests

**How to start:**

1. Pick an issue labeled `good-first` or `research`.
2. Reproduce baseline benchmarks.
3. Submit a small, well-tested PR.

---

## 📝 Changelog Highlights

* **v0.4.x:** Local stack stabilized; routing, safety, memory, demos; INT4 PTQ & BitNet scaffolds.
* **v0.5.0 (in progress):** INT4 kernels (CPU/DirectML), QAT adapters, on-device distillation, benchmark harness v1, plugin SDK.

---

## 🎯 Next Steps (Actionable)

**This week**

* Ship **INT4 CPU QGEMM** prototype + microbenchmarks
* QAT adapter training for a **3B chat** model at 4-bit
* Benchmark harness v1 (CLI + HTML report)

**Next 30 days**

* **DirectML INT4** path on AMD iGPU
* Idle-time **distillation loop** with reward-guided sampling
* Plugin SDK with **3 reference tools** and guardrails

**Quarter goal (Q4)**

* Public **results page** with reproducible runs
* **Student model** beats base 3B on user tasks, same latency
* **2× throughput** on 7B class with kernels

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🧩 The Bottom Line

* We turned “local LLMs on laptops” into a **real product**.
* Now we’re executing a **research-grade plan**: **custom low-bit kernels**, **QAT adapters**, **on-device distillation**, **reproducible benchmarks**, and a **safe plugin system**.
* This is how EdgeMind becomes both **useful today** and **novel tomorrow**.

**Start here:** `python src/core/edgemind.py --chat`
**Repo:** `github.com/artbyoscar/ai-playground`
