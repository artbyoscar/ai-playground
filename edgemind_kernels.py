"""EdgeMind High-Performance Kernels Python Interface"""
import ctypes
import numpy as np
from pathlib import Path
import platform

class EdgeMindKernels:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Auto-detect the library
            if platform.system() == "Windows":
                lib_path = Path(__file__).parent / "src/kernels/cpu/int4/build-final/qgemm_int4.dll"
            else:
                lib_path = Path(__file__).parent / "src/kernels/cpu/int4/build/libqgemm_int4.so"
        
        if not lib_path.exists():
            raise FileNotFoundError(f"EdgeMind kernel library not found at {lib_path}")
        
        self.lib = ctypes.CDLL(str(lib_path))
        self._setup_functions()
    
    def _setup_functions(self):
        # Setup Q8 GEMM function signature
        self.lib.qgemm_int4_fp16_q8_mt.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),  # A_fp16
            ctypes.c_int,                      # lda
            ctypes.POINTER(ctypes.c_int8),     # B_q8
            ctypes.POINTER(ctypes.c_uint16),   # scales
            ctypes.c_int,                      # ldb
            ctypes.POINTER(ctypes.c_uint16),   # C_fp16
            ctypes.c_int,                      # ldc
            ctypes.c_int,                      # M
            ctypes.c_int,                      # N
            ctypes.c_int,                      # K
            ctypes.c_int,                      # group_size
            ctypes.c_int,                      # num_threads
        ]
        self.lib.qgemm_int4_fp16_q8_mt.restype = None
    
    def q8_gemm(self, A, B_q8, scales, M, N, K, group_size=64, num_threads=8):
        """
        Perform Q8 quantized GEMM: C = A @ B
        
        Args:
            A: Input matrix (M x K) as float32, will be converted to fp16
            B_q8: Quantized weights (K x N) as int8
            scales: Per-group scales as fp16
            M, N, K: Matrix dimensions
            group_size: Quantization group size
            num_threads: Number of threads
        
        Returns:
            C: Output matrix (M x N) as float32
        """
        # Convert A to fp16
        A_fp16 = A.astype(np.float16).astype(np.uint16)
        
        # Prepare output
        C_fp16 = np.zeros((M, N), dtype=np.uint16)
        
        # Call kernel
        self.lib.qgemm_int4_fp16_q8_mt(
            A_fp16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            K,  # lda
            B_q8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            scales.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            0,  # ldb (unused)
            C_fp16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            N,  # ldc
            M, N, K, group_size, num_threads
        )
        
        # Convert back to float32
        return C_fp16.view(np.float16).astype(np.float32)

# Convenience function
def load_kernels():
    """Load EdgeMind kernels"""
    return EdgeMindKernels()
