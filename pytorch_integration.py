# pytorch_integration.py
import torch
import torch.nn as nn
from edgemind_kernels import q8_gemm

class EdgeMindLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_q8 = None  # Will hold quantized weights
        self.scales = None
        
    def forward(self, x):
        # Use your Q8 kernel
        return q8_gemm(x, self.weight_q8, self.scales)