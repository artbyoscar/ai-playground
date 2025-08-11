# src/core/gguf_loader.py
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import mmap

@dataclass
class GGUFTensor:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    offset: int
    size: int
    data: np.ndarray = None

class GGUFLoader:
    """Load GGUF models and prepare for EdgeMind kernels"""
    
    # GGUF magic number
    GGUF_MAGIC = 0x46554747  # "GGUF"
    
    # Quantization types
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.tensors: Dict[str, GGUFTensor] = {}
        self.metadata: Dict[str, Any] = {}
        self.file_handle = None
        self.mmap_handle = None
        
    def load(self):
        """Load GGUF model file"""
        print(f"Loading GGUF model: {self.model_path}")
        
        self.file_handle = open(self.model_path, 'rb')
        self.mmap_handle = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read header
        magic = struct.unpack('<I', self.mmap_handle[:4])[0]
        if magic != self.GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic: {magic:08x})")
        
        # Read version
        version = struct.unpack('<I', self.mmap_handle[4:8])[0]
        print(f"GGUF version: {version}")
        
        # Parse metadata and tensor info
        offset = 8
        offset = self._read_metadata(offset)
        offset = self._read_tensor_info(offset)
        
        print(f"Loaded {len(self.tensors)} tensors")
        return self
    
    def _read_metadata(self, offset: int) -> int:
        """Read model metadata"""
        # Simplified - real implementation needs full GGUF spec
        n_kv = struct.unpack('<Q', self.mmap_handle[offset:offset+8])[0]
        offset += 8
        
        for _ in range(n_kv):
            # Skip metadata for now (implement if needed)
            offset += 64  # Approximate
        
        return offset
    
    def _read_tensor_info(self, offset: int) -> int:
        """Read tensor information"""
        n_tensors = struct.unpack('<Q', self.mmap_handle[offset:offset+8])[0]
        offset += 8
        
        for i in range(n_tensors):
            # Read tensor name length
            name_len = struct.unpack('<Q', self.mmap_handle[offset:offset+8])[0]
            offset += 8
            
            # Read tensor name
            name = self.mmap_handle[offset:offset+name_len].decode('utf-8')
            offset += name_len
            
            # Read dimensions
            n_dims = struct.unpack('<I', self.mmap_handle[offset:offset+4])[0]
            offset += 4
            
            shape = []
            for _ in range(n_dims):
                dim = struct.unpack('<Q', self.mmap_handle[offset:offset+8])[0]
                shape.append(dim)
                offset += 8
            
            # Read type
            dtype = struct.unpack('<I', self.mmap_handle[offset:offset+4])[0]
            offset += 4
            
            # Read offset
            tensor_offset = struct.unpack('<Q', self.mmap_handle[offset:offset+8])[0]
            offset += 8
            
            # Store tensor info
            self.tensors[name] = GGUFTensor(
                name=name,
                shape=tuple(shape),
                dtype=dtype,
                offset=tensor_offset,
                size=np.prod(shape) * self._dtype_size(dtype)
            )
        
        return offset
    
    def get_tensor(self, name: str) -> np.ndarray:
        """Get tensor data as numpy array"""
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found")
        
        tensor = self.tensors[name]
        
        if tensor.data is None:
            # Load tensor data
            tensor.data = self._load_tensor_data(tensor)
        
        return tensor.data
    
    def _load_tensor_data(self, tensor: GGUFTensor) -> np.ndarray:
        """Load and dequantize tensor data"""
        # Seek to tensor data
        data_start = tensor.offset
        data_end = data_start + tensor.size
        raw_data = self.mmap_handle[data_start:data_end]
        
        # Dequantize based on type
        if tensor.dtype == self.GGML_TYPE_F32:
            return np.frombuffer(raw_data, dtype=np.float32).reshape(tensor.shape)
        elif tensor.dtype == self.GGML_TYPE_F16:
            return np.frombuffer(raw_data, dtype=np.float16).astype(np.float32).reshape(tensor.shape)
        elif tensor.dtype == self.GGML_TYPE_Q8_0:
            return self._dequantize_q8_0(raw_data, tensor.shape)
        elif tensor.dtype == self.GGML_TYPE_Q4_0:
            return self._dequantize_q4_0(raw_data, tensor.shape)
        else:
            raise NotImplementedError(f"Dtype {tensor.dtype} not supported yet")
    
    def _dequantize_q8_0(self, data: bytes, shape: Tuple) -> np.ndarray:
        """Dequantize Q8_0 format (matches your kernel format!)"""
        n_blocks = len(data) // 34  # 32 bytes quantized + 2 bytes scale
        output = np.zeros(n_blocks * 32, dtype=np.float32)
        
        for i in range(n_blocks):
            block_start = i * 34
            scale = np.frombuffer(data[block_start:block_start+2], dtype=np.float16)[0]
            quants = np.frombuffer(data[block_start+2:block_start+34], dtype=np.int8)
            
            # Dequantize
            output[i*32:(i+1)*32] = quants.astype(np.float32) * scale
        
        return output.reshape(shape)
    
    def _dtype_size(self, dtype: int) -> int:
        """Get size in bytes for dtype"""
        sizes = {
            self.GGML_TYPE_F32: 4,
            self.GGML_TYPE_F16: 2,
            self.GGML_TYPE_Q8_0: 1.0625,  # 34 bytes per 32 values
            self.GGML_TYPE_Q4_0: 0.5625,  # 18 bytes per 32 values
        }
        return sizes.get(dtype, 4)
    
    def prepare_for_edgemind(self, tensor_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare tensor for EdgeMind kernels
        Returns: (quantized_data, scales)
        """
        tensor_data = self.get_tensor(tensor_name)
        
        # If already quantized in compatible format
        if self.tensors[tensor_name].dtype == self.GGML_TYPE_Q8_0:
            # Already in Q8 format - just return
            # (This is simplified - need proper extraction)
            return tensor_data, np.ones(tensor_data.shape[0] // 32)
        
        # Otherwise quantize to our format
        from tools.quant.quantize_q8_edge import quantize_q8_symmetric
        return quantize_q8_symmetric(tensor_data, group_size=64)