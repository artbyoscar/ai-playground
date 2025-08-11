#!/usr/bin/env python3
"""Q8 quantizer for EdgeMind kernels."""
import numpy as np
import json
import argparse
from pathlib import Path

def quantize_q8_symmetric(weights, group_size=64):
    """Quantize to int8 with symmetric per-group scaling."""
    K, N = weights.shape
    groups = (K + group_size - 1) // group_size
    
    quantized = np.zeros((K, N), dtype=np.int8)
    scales = np.zeros((groups, N), dtype=np.float16)
    
    for n in range(N):
        col = weights[:, n]
        for g in range(groups):
            k_start = g * group_size
            k_end = min(k_start + group_size, K)
            group_vals = col[k_start:k_end]
            
            max_abs = np.max(np.abs(group_vals))
            if max_abs > 1e-6:
                scale = max_abs / 127.0
                inv_scale = 1.0 / scale
                q_vals = np.round(group_vals * inv_scale).astype(np.int32)
                q_vals = np.clip(q_vals, -127, 127)
                quantized[k_start:k_end, n] = q_vals.astype(np.int8)
            else:
                scale = 1.0
                quantized[k_start:k_end, n] = 0
            
            scales[g, n] = np.float16(scale)
    
    return quantized, scales

def dequantize_q8(quantized, scales, group_size=64):
    """Dequantize Q8 back to float32."""
    K, N = quantized.shape
    groups = scales.shape[0]
    dequantized = np.zeros((K, N), dtype=np.float32)
    
    for n in range(N):
        for g in range(groups):
            k_start = g * group_size
            k_end = min(k_start + group_size, K)
            scale = scales[g, n].astype(np.float32)
            dequantized[k_start:k_end, n] = quantized[k_start:k_end, n].astype(np.float32) * scale
    
    return dequantized

def main():
    parser = argparse.ArgumentParser(description="Q8 quantizer")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    
    B = np.load(args.weights).astype(np.float32)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    K, N = B.shape
    print(f"Shape: K={K}, N={N}")
    
    quantized, scales = quantize_q8_symmetric(B, args.group)
    
    packed_data = []
    packed_scales = []
    for n in range(N):
        packed_data.append(quantized[:, n])
        packed_scales.append(scales[:, n])
    
    packed_data = np.concatenate(packed_data)
    packed_scales = np.concatenate(packed_scales)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    packed_data.tofile(f"{args.out}.q8.bin")
    packed_scales.astype(np.uint16).tofile(f"{args.out}.scales.fp16.bin")
    
    metadata = {
        "format": "q8_edge",
        "K": K,
        "N": N,
        "group_size": args.group,
        "data_file": f"{out_path.name}.q8.bin",
        "scales_file": f"{out_path.name}.scales.fp16.bin"
    }
    
    with open(f"{args.out}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved: {args.out}.{{q8.bin, scales.fp16.bin, json}}")
    
    if args.verify:
        # Properly dequantize and compute error
        dequantized = dequantize_q8(quantized, scales, args.group)
        error = np.linalg.norm(B - dequantized) / np.linalg.norm(B)
        print(f"Relative error: {error:.4e}")
        
        # Show some examples
        print("\nSample values (original -> quantized -> dequantized):")
        for _ in range(5):
            i = np.random.randint(0, K)
            j = np.random.randint(0, N)
            print(f"  [{i},{j}]: {B[i,j]:.4f} -> {quantized[i,j]:4d} -> {dequantized[i,j]:.4f}")

if __name__ == "__main__":
    main()
