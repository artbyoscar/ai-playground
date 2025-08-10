EdgeMind Kernels — INT4 CPU (AVX2 + F16C)
A focused workstream to make low-bit CPU inference actually fast on consumer laptops. This folder contains a working INT4 QGEMM prototype (weights in INT4 with per-group FP16 scales, activations in FP16), tests, micro-benchmarks, and simple quantization utilities.

What’s new (this update)
✅ Correctness harness passing at --threshold 7.2e-2 across multiple shapes (st/mt/tmt all pass).

🧰 Packed loader + perf harness

pack_loader.{h,cpp} to read tools/quant outputs.

tests/test_perf_load_packed.cpp to run kernels directly on disk-packed B.

🧪 Quant packers + error check (tools/quant): rel_err ≈ 5.71e-2 for a random K=2048, N=256 with group=64.

🧱 DML subtree scaffold (dml/int4) with a smoke test wired into CTest.

🧵 Windows build reliability: Ninja via pip install ninja; CMake/ctest on PATH; PowerShell quirks documented.

Requirements
CPU with AVX2 + F16C.

LLVM/Clang (tested: 20.1.8), CMake ≥ 3.22, Ninja.

Python 3.13, NumPy (for the packers).

PowerShell note: the python - <<'PY' heredoc trick isn’t supported—put snippets in a .py file and run them.

Build & test (Windows, PowerShell)
From src/kernels/cpu/int4:

powershell
Copy code
# One-time: ensure tools are visible in the shell
$env:Path = "C:\Program Files\LLVM\bin;C:\Program Files\CMake\bin;$env:AppData\Python\Python313\Scripts;$env:Path"

# Generate a random test matrix
python .\B_gen.py

# Quantize column-wise (group=64), write {bin, fp16 scales, json}
python ..\..\..\..\tools\quant\quantize_q4_edge.py --weights .\B.npy --out .\pack\q4edge --group 64

# Configure + build
Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release

# Correctness (current threshold = 7.2e-2)
ctest --test-dir .\build -R correctness -V

# Micro-benchmarks
.\build\test_qgemm_perf_vs_baseline.exe --M 256 --N 256 --K 2048 --it 5
.\build\test_perf_load_packed.exe --packed .\pack\q4edge --M 256 --it 5
Expected outputs (representative)
bash
Copy code
# correctness
rel ≈ 6.6e-2 … 7.1e-2  → PASS at 7.2e-2

# perf vs FP32 (latest run, M=256 N=256 K=2048 it=5)
FP32 ref: ~235 ms (≈1.14 GFLOP/s)
st  : ~21.0 ms (≈12.8 GFLOP/s)   ~11.2×
mt  :  ~4.68 ms (≈57.3 GFLOP/s)  ~50.2×
tmt :  ~4.42 ms (≈60.7 GFLOP/s)  ~53.2×
tmt+m: ~4.32 ms (≈62.2 GFLOP/s)  ~54.5×  (prototype M-blocked tiled path)

# perf vs packed B (loader path)
st  : ~18.8 ms (≈14.3 GFLOP/s)
mt  :  ~4.03 ms (≈66.7 GFLOP/s)
tmt :  ~3.66 ms (≈73.4 GFLOP/s)
Perf varies with power/thermals; expect ±10–20%. Regenerate locally to compare.

Quantization tools (PTQ prototype)
powershell
Copy code
# Create a random test matrix (K=2048, N=256)
python .\B_gen.py

# Quantize column-wise with group=64
python ..\..\..\..\tools\quant\quantize_q4_edge.py --weights .\B.npy --out .\pack\q4edge --group 64

# Decode + compare error vs FP32 (proxy via Bᵀ·B)
python ..\..\..\..\tools\quant\check_error.py --weights .\B.npy --pack .\pack\q4edge --group 64
# max_abs≈4.56e+00  mean_abs≈8.39e-01  rel_err≈5.71e-02
Layout used by the kernel

Column-wise K blocks; group_size = 64; bytes_per_group = 32 (two nibbles/byte).

Per-group FP16 scale (saved as raw uint16).

What we learned (so far)
Earlier correctness failures were packing order + FP16 conversion; fixed by:

Packing per column along K in 64-sized groups.

Precise scalar FP16↔FP32 fallback (F16C fast path when available).

AVX2 nibble-unpack that exactly matches the encoding.

Tiling to reuse A moves the needle more than raw prefetch on laptop CPUs.

NumPy packers are gold for quick math sanity without touching C++.

Next steps (near-term)
Tighten threshold to 7e-2 after one more pass on packing/accum rounding; keep 7.2e-2 in CI for safety.

Promote M-blocked kernel (qgemm_int4_fp16_tiled_mt_mblocked) to a first-class benchmark config with CLI flags:

--threads, --nc-tile, --m-tile; print settings with results and emit CSV.

Epilogue fusion: bias + ReLU/SILU/GELU (tanh) to cut bandwidth and improve E2E latency.

Prefetch polish: prefetch A ~128B ahead when K - k ≥ 128; keep B prefetch tight to column tile.

Pack format doc + validator: tiny tool to verify K/N/group/byte-count match the JSON before running.

CI: GitHub Actions (Windows runner) that builds and runs ctest -R correctness; optional artifact with perf CSV.

DML backend: replace smoke test with a real DirectML path that consumes the same pack format.

DML subtree (scaffold)
src/kernels/cpu/int4/dml/int4 includes a tiny CTest smoke target (dml_qgemm_smoke). It currently verifies test wiring only. Real DML kernels + packers land next milestone.

Troubleshooting
ctest/cmake not found → add C:\Program Files\CMake\bin to PATH.

ninja not found → python -m pip install ninja and add %AppData%\Python\Python313\Scripts to PATH.

PowerShell heredoc (python - <<'PY') won’t work → save snippet to a .py and run.

Suppress fopen deprecation (clang-cl on Windows):

Add at top of pack_loader.cpp: #define _CRT_SECURE_NO_WARNINGS

Or globally in CMake: add_definitions(-D_CRT_SECURE_NO_WARNINGS)

Changelog (snippet)
2025-08-10

Correctness passing at 7.2e-2; rel ~6.6–7.1e-2 across shapes.

Added pack_loader + test_perf_load_packed; loader path perf validated.

Perf bench updated (baseline + packed); best tmt ≈ 60–74 GFLOP/s depending on power state.

Quant tools stabilized; rel_err ~5.7e-2 on random K=2048, N=256.

DML smoke test via CTest.

Windows build instructions + PATH tips.

License: MIT

