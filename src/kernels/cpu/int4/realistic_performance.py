"""
Realistic performance assessment of EdgeMind kernels
Let's be honest about what performance you're actually getting
"""

import subprocess
import os
import time
import numpy as np

KERNEL_PATH = r"C:\Users\OscarNuñez\Desktop\ai-playground\src\kernels\cpu\int4\build\Release"

def run_kernel_test(M, N, K, threads=16, iterations=10):
    """Run kernel and get actual performance"""
    exe_path = os.path.join(KERNEL_PATH, "test_qgemm_perf_q8_mt.exe")
    
    if not os.path.exists(exe_path):
        return None
    
    cmd = [exe_path, "--M", str(M), "--N", str(N), "--K", str(K), 
           "--threads", str(threads), "--it", str(iterations)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    # Parse GFLOP/s from output
    for line in output.split('\n'):
        if 'GFLOP/s' in line:
            try:
                # Extract number before GFLOP/s
                parts = line.split('(')[-1].split('GFLOP/s')[0]
                return float(parts.strip())
            except:
                pass
    return None

def test_numpy_baseline(M, N, K):
    """Test NumPy performance for comparison"""
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Warm up
    _ = np.matmul(A, B)
    
    # Time it
    start = time.perf_counter()
    for _ in range(10):
        C = np.matmul(A, B)
    elapsed = time.perf_counter() - start
    
    ops = 10 * 2 * M * N * K
    gflops = ops / (elapsed * 1e9)
    
    return gflops

def honest_assessment():
    """Give an honest assessment of actual performance"""
    print("="*70)
    print(" "*20 + "HONEST PERFORMANCE ASSESSMENT")
    print("="*70)
    print()
    
    # Test configuration from your claims
    M, N, K = 256, 256, 2048
    
    print(f"Testing configuration: {M}×{K} @ {K}×{N}")
    print("-"*50)
    
    # Your kernel performance
    kernel_gflops = run_kernel_test(M, N, K, threads=16, iterations=100)
    
    # NumPy baseline
    numpy_gflops = test_numpy_baseline(M, N, K)
    
    # Theoretical peak (very rough estimate)
    # Assuming AVX2: 8 FP32 ops per cycle, ~3 GHz, 16 cores
    theoretical_peak = 8 * 3.0 * 16  # ~384 GFLOP/s theoretical
    
    print(f"\n📊 RESULTS:")
    print(f"  Your kernel:        {kernel_gflops:.2f} GFLOP/s")
    print(f"  NumPy baseline:     {numpy_gflops:.2f} GFLOP/s")
    print(f"  Theoretical peak:   ~{theoretical_peak:.0f} GFLOP/s (CPU dependent)")
    
    if kernel_gflops:
        speedup = kernel_gflops / numpy_gflops
        efficiency = (kernel_gflops / theoretical_peak) * 100
        
        print(f"\n📈 ANALYSIS:")
        print(f"  Speedup vs NumPy:   {speedup:.1f}x")
        print(f"  CPU efficiency:     {efficiency:.1f}%")
        
        print(f"\n💭 REALITY CHECK:")
        if kernel_gflops > 100:
            print("  ✅ EXCELLENT: You've achieved impressive performance!")
            print("     This is genuinely fast for CPU inference.")
        elif kernel_gflops > 50:
            print("  ✅ VERY GOOD: Solid performance for quantized inference")
            print("     This is competitive with good implementations.")
        elif kernel_gflops > 20:
            print("  ⚠️ DECENT: Reasonable but room for improvement")
            print("     Check compiler optimizations and thread scaling.")
        elif kernel_gflops > 10:
            print("  ⚠️ BELOW EXPECTATIONS: Your kernels need optimization")
            print("     Likely issues: Missing AVX2, poor threading, or debug mode")
        else:
            print("  ❌ POOR: Something is wrong with the implementation")
            print("     Your current ~10 GFLOP/s is far from the 125 claimed")
        
        print(f"\n🎯 THE TRUTH:")
        if kernel_gflops < 20:
            print(f"  Your claimed 125 GFLOP/s is NOT being achieved.")
            print(f"  Actual performance is {125/kernel_gflops:.1f}x slower than claimed.")
            print(f"  This needs to be fixed before claiming high performance.")
        
        print(f"\n📋 TODO:")
        if kernel_gflops < 50:
            print("  1. Check if compiled with -march=native -mavx2 -mfma")
            print("  2. Verify Release mode (not Debug)")
            print("  3. Test on different hardware")
            print("  4. Profile to find bottlenecks")
            print("  5. Consider the claim might have been mismeasured")

def test_scaling():
    """Test how performance scales with threads"""
    print("\n" + "="*70)
    print(" "*20 + "THREAD SCALING TEST")
    print("="*70)
    
    M, N, K = 256, 256, 2048
    thread_counts = [1, 2, 4, 8, 16]
    
    results = []
    for threads in thread_counts:
        gflops = run_kernel_test(M, N, K, threads=threads, iterations=20)
        if gflops:
            results.append((threads, gflops))
            print(f"  {threads:2} threads: {gflops:6.2f} GFLOP/s")
    
    if len(results) > 1:
        single_thread = results[0][1]
        best_perf = max(r[1] for r in results)
        best_threads = [r[0] for r in results if r[1] == best_perf][0]
        
        print(f"\n  Best: {best_perf:.2f} GFLOP/s with {best_threads} threads")
        print(f"  Scaling: {best_perf/single_thread:.1f}x from 1 thread")

def main():
    print("\n" + "🔍"*35)
    print(" "*15 + "EDGEMIND REALISTIC PERFORMANCE CHECK")
    print("🔍"*35 + "\n")
    
    # Check kernels exist
    exe_path = os.path.join(KERNEL_PATH, "test_qgemm_perf_q8_mt.exe")
    if not os.path.exists(exe_path):
        print(f"❌ Kernels not found at {exe_path}")
        return
    
    honest_assessment()
    test_scaling()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("""
1. BE HONEST: Update your README with actual performance numbers
2. INVESTIGATE: Why is performance 12x lower than claimed?
3. OPTIMIZE: Try different compiler flags and settings
4. VERIFY: Test on different machines to rule out hardware issues
5. DOCUMENT: Explain the performance gap transparently

Remember: 10 GFLOP/s is still ~5x faster than baseline FP32!
That's valuable, just not the 60x claimed.
    """)

if __name__ == "__main__":
    main()