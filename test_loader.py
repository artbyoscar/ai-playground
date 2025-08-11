# test_loader.py
from src.core.gguf_loader import GGUFLoader

# Download a small model for testing
# Run: ollama pull phi3:mini
# Find the model file (usually in ~/.ollama/models/)

model_path = "path/to/phi-3-mini.gguf"  # Update this path
loader = GGUFLoader(model_path)
loader.load()

# List all tensors
print("Model tensors:")
for name, tensor in loader.tensors.items():
    print(f"  {name}: {tensor.shape} ({tensor.dtype})")

# Load a weight tensor
weight = loader.get_tensor("model.layers.0.self_attn.q_proj.weight")
print(f"Weight shape: {weight.shape}")

# Prepare for your kernels
quantized, scales = loader.prepare_for_edgemind("model.layers.0.self_attn.q_proj.weight")
print(f"Quantized shape: {quantized.shape}")
print(f"Scales shape: {scales.shape}")