# test_components.py
import sys
import os

print("Testing component imports...")

# Test 1: BitNet
try:
    from src.models.bitnet import BitLinear
    print("✅ BitNet works!")
except Exception as e:
    print(f"❌ BitNet failed: {e}")

# Test 2: EdgeFormer
try:
    from src.optimization.utils.quantization import Int4Quantizer
    print("✅ EdgeFormer works!")
except Exception as e:
    print(f"❌ EdgeFormer failed: {e}")

# Test 3: HybridCompute
try:
    from src.compute.hybrid_compute_manager import HybridComputeManager
    print("✅ HybridCompute works!")
except Exception as e:
    print(f"❌ HybridCompute failed: {e}")