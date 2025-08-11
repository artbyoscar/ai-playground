"""
Simplified GGUF loader - just enough to read weights
No sentencepiece or transformers needed!
"""

import struct
import numpy as np
from pathlib import Path
import mmap

class SimpleGGUFLoader:
    """Minimal GGUF loader - just reads tensor data"""
    
    GGUF_MAGIC = 0x46554747  # "GGUF"
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.file = None
        self.mmap = None
        
    def open(self):
        """Open and memory-map the file"""
        self.file = open(self.model_path, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Check magic
        magic = struct.unpack('<I', self.mmap[:4])[0]
        if magic != self.GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file!")
        
        print(f"✅ Opened GGUF file: {self.model_path.name}")
        return self
    
    def read_tensor_at_offset(self, offset: int, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Read tensor data at specific offset"""
        elements = np.prod(shape)
        bytes_needed = elements * dtype().itemsize
        
        data = self.mmap[offset:offset + bytes_needed]
        tensor = np.frombuffer(data, dtype=dtype).reshape(shape)
        return tensor
    
    def find_tensor_simple(self, name_pattern: str):
        """Super simple tensor finder - just scans for the name"""
        # This is a hack but works for testing
        pattern = name_pattern.encode('utf-8')
        idx = self.mmap.find(pattern)
        
        if idx != -1:
            print(f"Found pattern '{name_pattern}' at offset {idx}")
            # Tensor data usually starts a bit after the name
            # This is very approximate!
            return idx + len(pattern) + 100  # Skip metadata
        return None
    
    def close(self):
        """Clean up"""
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()


def test_loader():
    """Test the loader with a real model"""
    
    # First, check what models we have
    model_dir = Path("C:/EdgeMindModels")  # Where we moved them
    if not model_dir.exists():
        model_dir = Path("models")  # Or original location
    
    gguf_files = list(model_dir.glob("*.gguf"))
    
    if not gguf_files:
        print("❌ No GGUF files found!")
        print("Download one with: ollama pull phi3:mini")
        return
    
    print(f"Found {len(gguf_files)} GGUF files:")
    for i, f in enumerate(gguf_files):
        size_gb = f.stat().st_size / (1024**3)
        print(f"  {i}: {f.name} ({size_gb:.2f} GB)")
    
    # Use the smallest one
    smallest = min(gguf_files, key=lambda f: f.stat().st_size)
    print(f"\n📦 Using smallest: {smallest.name}")
    
    # Open and test
    loader = SimpleGGUFLoader(smallest)
    loader.open()
    
    # Try to find a tensor
    offset = loader.find_tensor_simple("attn.q_proj")
    if offset:
        # Read some data (just as a test)
        data = loader.read_tensor_at_offset(offset, (32, 32), np.float16)
        print(f"Read tensor shape: {data.shape}")
        print(f"Tensor stats: min={data.min():.3f}, max={data.max():.3f}, mean={data.mean():.3f}")
    
    loader.close()
    print("✅ Loader test complete!")


if __name__ == "__main__":
    test_loader()