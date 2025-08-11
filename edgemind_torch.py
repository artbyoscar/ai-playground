"""PyTorch integration for EdgeMind kernels"""
import torch
import torch.nn as nn
import numpy as np
from edgemind_kernels import load_kernels

class EdgeMindLinear(nn.Module):
    """Drop-in replacement for nn.Linear with Q8 quantization"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize with normal Linear layer
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        
        # Quantization placeholders
        self.weight_q8 = None
        self.scales = None
        self.kernels = load_kernels()
    
    def quantize_weights(self, group_size=64):
        """Quantize weights to INT8"""
        import sys
        sys.path.append("tools/quant")
        from quantize_q8_edge import quantize_q8_symmetric
        
        W = self.weight.detach().cpu().numpy()
        self.weight_q8, self.scales = quantize_q8_symmetric(W.T, group_size)
        
        # Pack for kernel
        self.weight_q8_packed = np.concatenate([self.weight_q8[:, n] for n in range(W.shape[0])])
        self.scales_packed = np.concatenate([self.scales[:, n] for n in range(W.shape[0])])
        
        # Free original weights
        self.weight = None
    
    def forward(self, x):
        if self.weight_q8 is not None:
            # Use quantized kernel
            batch_size = x.shape[0]
            x_np = x.detach().cpu().numpy().reshape(batch_size, -1)
            
            output = self.kernels.q8_gemm(
                x_np, 
                self.weight_q8_packed,
                self.scales_packed.astype(np.uint16),
                batch_size,
                self.out_features,
                self.in_features,
                64, 8
            )
            
            output = torch.from_numpy(output).to(x.device)
            if self.bias is not None:
                output += self.bias
            return output
        else:
            # Fallback to normal linear
            return F.linear(x, self.weight, self.bias)

# Example usage
if __name__ == "__main__":
    # Create model
    model = nn.Sequential(
        EdgeMindLinear(2048, 512),
        nn.ReLU(),
        EdgeMindLinear(512, 256)
    )
    
    # Quantize
    for layer in model:
        if isinstance(layer, EdgeMindLinear):
            layer.quantize_weights()
    
    # Test
    x = torch.randn(32, 2048)
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
