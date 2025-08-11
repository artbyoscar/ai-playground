"""
Ultra simple test - no numpy, no dependencies
Just proves the kernel module loads
"""

import sys
import os
import time

# Add paths where module might be
paths_to_try = [
    '.',
    'src/kernels/cpu/int4',
    'src\\kernels\\cpu\\int4',
    'build',
    'build\\Release'
]

for path in paths_to_try:
    if os.path.exists(path):
        sys.path.insert(0, path)

# Try to import
try:
    import edgemind_core
    print("✅ SUCCESS! EdgeMind module loaded!")
    print(f"   Available functions: {dir(edgemind_core)}")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    print("\nTo fix:")
    print("1. cd src\\kernels\\cpu\\int4")
    print("2. build_ultra_simple.bat")
    sys.exit(1)

# Test the kernel
print("\n" + "="*50)
print("Testing EdgeMind Kernels")
print("="*50)

test_sizes = [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
]

for M, N, K in test_sizes:
    print(f"\nTest {M}x{K} * {K}x{N} = {M}x{N}")
    
    try:
        # Warm up
        result = edgemind_core.test_kernel(M, N, K)
        
        # Time it
        start = time.perf_counter()
        for _ in range(10):
            result = edgemind_core.test_kernel(M, N, K)
        elapsed = time.perf_counter() - start
        
        # Calculate approximate GFLOPS
        flops_per_call = 2 * M * N * K
        total_flops = flops_per_call * 10
        gflops = total_flops / (elapsed * 1e9)
        
        print(f"  Result: {result}")
        print(f"  Time for 10 calls: {elapsed*1000:.2f} ms")
        print(f"  Approx performance: {gflops:.2f} GFLOP/s")
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*50)
print("If you see results above, your kernels are accessible!")
print("Next step: Connect to real model weights")
print("="*50)