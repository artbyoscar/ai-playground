# Create pybind11 wrapper for your kernels
# edgemind_kernels.py
import ctypes
import numpy as np

class EdgeMindKernels:
    def __init__(self, lib_path="./build-final/qgemm_int4.dll"):
        self.lib = ctypes.CDLL(lib_path)
        # Setup function signatures...
    
    def gemm_q8(self, A, B_packed, scales):
        # Call your Q8 kernel
        pass