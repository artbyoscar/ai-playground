import subprocess
import os

# Simple test script - no path issues
os.chdir("C:/Users/OscarNuñez/Desktop/ai-playground/src/kernels/cpu/int4")

print("Testing EdgeMind Kernel Performance")
print("=" * 50)

# Test different configurations
configs = [
    (256, 256, 2048, 1),
    (256, 256, 2048, 2),
    (256, 256, 2048, 4),
    (256, 256, 2048, 8),
    (512, 512, 2048, 8),
    (128, 128, 1024, 8),
]

best = 0
best_config = ""

for M, N, K, threads in configs:
    cmd = f".\\build-final\\test_qgemm_perf_q8_mt.exe --M {M} --N {N} --K {K} --threads {threads} --it 10"
    print(f"\nTesting: M={M} N={N} K={K} threads={threads}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout
    print(output)
    
    # Find GFLOP/s in output
    for line in output.split('\n'):
        if 'GFLOP/s' in line:
            try:
                # Extract the number
                import re
                match = re.search(r'\(([0-9.]+)\s*GFLOP/s\)', line)
                if match:
                    gflops = float(match.group(1))
                    if gflops > best:
                        best = gflops
                        best_config = f"M={M} N={N} K={K} threads={threads}"
            except:
                pass

print("\n" + "=" * 50)
print(f"Best performance: {best:.2f} GFLOP/s")
print(f"Configuration: {best_config}")
print(f"Speedup vs FP32 (2.1 GFLOP/s): {best/2.1:.1f}x")