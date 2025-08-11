"""Test EdgeMind Kernels Integration"""
import numpy as np
import time
from pathlib import Path
import sys

# Add tools to path
sys.path.append(str(Path(__file__).parent / "tools/quant"))

def test_q8_kernel():
    """Test Q8 kernel performance"""
    print("🚀 Testing EdgeMind Q8 Kernel")
    print("=" * 40)
    
    # Test dimensions
    M, N, K = 256, 256, 2048
    group_size = 64
    
    # Generate random data
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Quantize B using your Python quantizer
    from quantize_q8_edge import quantize_q8_symmetric
    B_q8, scales = quantize_q8_symmetric(B, group_size)
    
    # Pack for kernel (column-wise)
    B_q8_packed = np.concatenate([B_q8[:, n] for n in range(N)])
    scales_packed = np.concatenate([scales[:, n] for n in range(N)])
    
    print(f"Matrix dimensions: {M}×{N}×{K}")
    print(f"Quantization group size: {group_size}")
    print(f"B quantized to INT8 with {scales.shape[0]} groups")
    
    # Load kernel (this would work if we had the DLL)
    try:
        from edgemind_kernels import load_kernels
        kernels = load_kernels()
        
        # Warm-up
        _ = kernels.q8_gemm(A, B_q8_packed, scales_packed.astype(np.uint16), M, N, K, group_size, 8)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            C = kernels.q8_gemm(A, B_q8_packed, scales_packed.astype(np.uint16), M, N, K, group_size, 8)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        gflops = (2.0 * M * N * K) / (avg_time / 1000) / 1e9
        
        print(f"\n✅ Kernel Performance:")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  Performance: {gflops:.2f} GFLOP/s")
        
        # Verify correctness
        C_ref = A @ B
        error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
        print(f"  Error vs FP32: {error:.4e}")
        
    except (ImportError, FileNotFoundError) as e:
        print(f"\n⚠️  Kernel library not available: {e}")
        print("  Build the DLL first with:")
        print("  cd src/kernels/cpu/int4")
        print("  cmake --build build-final --config Release")
    
    # Test Python quantization
    print(f"\n📊 Quantization Stats:")
    print(f"  Original range: [{B.min():.3f}, {B.max():.3f}]")
    print(f"  Quantized range: [{B_q8.min()}, {B_q8.max()}]")
    print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]")

if __name__ == "__main__":
    test_q8_kernel()
