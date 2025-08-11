"""
Simple test for EdgeMind kernels - minimal dependencies
Run from project root after building
"""

import sys
import os
import time
import struct
import random

# Try to import the module
try:
    # Try different paths
    possible_paths = [
        '.',
        'src/kernels/cpu/int4',
        'build',
        'build/Release'
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'edgemind_core.pyd')):
            sys.path.insert(0, path)
            break
    
    import edgemind_core
    print("✅ EdgeMind module loaded!")
    print(f"Available functions: {[x for x in dir(edgemind_core) if not x.startswith('_')]}")
    
except ImportError as e:
    print(f"❌ Could not load edgemind_core: {e}")
    print("\nTo fix:")
    print("1. cd src/kernels/cpu/int4")
    print("2. powershell .\\build_simple.ps1")
    sys.exit(1)

def create_test_data(M, N, K):
    """Create test matrices as bytes"""
    # Create random float32 matrix A (M x K)
    A = []
    for _ in range(M * K):
        A.append(random.uniform(-1, 1))
    A_bytes = struct.pack(f'{M*K}f', *A)
    
    # Create random int8 matrix B (K x N)
    B = []
    for _ in range(K * N):
        B.append(random.randint(-127, 127))
    B_bytes = struct.pack(f'{K*N}b', *B)
    
    # Create scales (simplified - one per output column)
    scales = []
    num_groups = (K * N + 63) // 64  # group_size = 64
    for _ in range(num_groups):
        scales.append(0.01)  # Simple scale
    scales_bytes = struct.pack(f'{num_groups}f', *scales)
    
    return A_bytes, B_bytes, scales_bytes

def test_kernel_performance():
    """Test kernel with different sizes"""
    print("\n" + "="*60)
    print("🚀 Testing EdgeMind Kernel Performance")
    print("="*60)
    
    test_sizes = [
        (32, 32, 32),      # Tiny
        (128, 128, 128),   # Small
        (256, 256, 256),   # Medium
        (256, 256, 2048),  # Your benchmark size
    ]
    
    for M, N, K in test_sizes:
        print(f"\n📊 Testing {M}x{K} @ {K}x{N} -> {M}x{N}")
        
        # Create test data
        A_bytes, B_bytes, scales_bytes = create_test_data(M, N, K)
        
        try:
            # Warm up
            _ = edgemind_core.q8_gemm_raw(A_bytes, B_bytes, scales_bytes, M, N, K)
            
            # Time the kernel
            start = time.perf_counter()
            C_bytes = edgemind_core.q8_gemm_raw(A_bytes, B_bytes, scales_bytes, M, N, K)
            elapsed = time.perf_counter() - start
            
            # Calculate performance
            flops = 2 * M * N * K
            gflops = flops / (elapsed * 1e9)
            
            print(f"  ✅ Success!")
            print(f"  ⚡ Performance: {gflops:.2f} GFLOP/s")
            print(f"  ⏱️ Time: {elapsed*1000:.2f} ms")
            print(f"  📦 Output size: {len(C_bytes)} bytes (expected: {M*N*4})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_with_numpy():
    """Test with numpy if available"""
    try:
        import numpy as np
        print("\n" + "="*60)
        print("🔬 Testing with NumPy comparison")
        print("="*60)
        
        M, N, K = 256, 256, 512
        
        # Create numpy arrays
        A = np.random.randn(M, K).astype(np.float32)
        B_float = np.random.randn(K, N).astype(np.float32)
        
        # Quantize B
        B_q8 = np.clip(B_float * 127 / np.abs(B_float).max(), -127, 127).astype(np.int8)
        scales = np.ones((K * N + 63) // 64, dtype=np.float32) * (np.abs(B_float).max() / 127)
        
        # Convert to bytes
        A_bytes = A.tobytes()
        B_bytes = B_q8.tobytes()
        scales_bytes = scales.tobytes()
        
        # Run kernel
        print(f"Testing {M}x{K} @ {K}x{N}")
        start = time.perf_counter()
        C_bytes = edgemind_core.q8_gemm_raw(A_bytes, B_bytes, scales_bytes, M, N, K)
        kernel_time = time.perf_counter() - start
        
        # Convert result back
        C_kernel = np.frombuffer(C_bytes, dtype=np.float32).reshape(M, N)
        
        # Compare with numpy
        start = time.perf_counter()
        C_numpy = np.matmul(A, B_float)
        numpy_time = time.perf_counter() - start
        
        # Results
        kernel_gflops = (2 * M * N * K) / (kernel_time * 1e9)
        numpy_gflops = (2 * M * N * K) / (numpy_time * 1e9)
        
        print(f"\n📊 Results:")
        print(f"  EdgeMind: {kernel_gflops:.2f} GFLOP/s ({kernel_time*1000:.2f} ms)")
        print(f"  NumPy:    {numpy_gflops:.2f} GFLOP/s ({numpy_time*1000:.2f} ms)")
        print(f"  Speedup:  {kernel_gflops/numpy_gflops:.1f}x")
        
        # Check accuracy (will be approximate due to quantization)
        error = np.mean(np.abs(C_kernel - C_numpy))
        rel_error = error / np.mean(np.abs(C_numpy))
        print(f"  Relative error: {rel_error:.2%} (expected due to quantization)")
        
    except ImportError:
        print("\n📝 NumPy not available, skipping comparison")

def main():
    print("\n" + "="*70)
    print(" "*20 + "EdgeMind Kernel Test Suite")
    print("="*70)
    
    # Test basic functionality
    test_kernel_performance()
    
    # Test with numpy if available
    test_with_numpy()
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("\nNext steps:")
    print("1. If performance is good (>50 GFLOP/s), kernels are working!")
    print("2. Connect to real GGUF model weights")
    print("3. Build full inference pipeline")
    print("="*70)

if __name__ == "__main__":
    main()