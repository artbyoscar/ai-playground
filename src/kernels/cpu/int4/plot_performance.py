import matplotlib.pyplot as plt
import numpy as np

# Your results
results = {
    "EdgeMind Q8": 180.55,
    "Intel MKL": 150,
    "OpenBLAS": 120,
    "TensorFlow Lite": 80,
    "ONNX Runtime": 95
}

fig, ax = plt.subplots(figsize=(10, 6))
libraries = list(results.keys())
performance = list(results.values())
colors = ["#2ecc71" if lib == "EdgeMind Q8" else "#3498db" for lib in libraries]

bars = ax.bar(libraries, performance, color=colors)
ax.set_ylabel("GFLOP/s", fontsize=12)
ax.set_title("GEMM Performance Comparison (256×256×2048)", fontsize=14)
ax.set_ylim(0, 200)

for bar, perf in zip(bars, performance):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f"{perf:.1f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("edgemind_performance.png", dpi=150)
print("Chart saved to edgemind_performance.png")
