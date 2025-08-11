// src/kernels/cpu/int4/model_interface.h
#ifndef EDGEMIND_MODEL_INTERFACE_H
#define EDGEMIND_MODEL_INTERFACE_H

#include <cstdint>
#include <vector>

namespace edgemind {

// Tensor structure that matches model weights
struct Tensor {
    float* data_fp32;      // Original weights
    int8_t* data_int8;     // Quantized weights
    float* scales;         // Quantization scales
    int dims[4];          // [batch, seq_len, hidden, heads]
    size_t numel;         // Total elements
    bool quantized;       // Is this quantized?
};

// Layer definition
struct LinearLayer {
    Tensor weight;
    Tensor bias;
    int in_features;
    int out_features;
};

// Model forward pass interface
class ModelKernel {
public:
    // Initialize with your existing kernels
    ModelKernel();
    
    // Forward pass through linear layer using YOUR kernels
    Tensor forward_linear(const Tensor& input, const LinearLayer& layer);
    
    // Attention using YOUR kernels
    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v);
    
    // Your existing GEMM kernel wrapped
    void gemm_int8(
        const float* A, 
        const int8_t* B_quant,
        const float* scales,
        float* C,
        int M, int N, int K
    );
    
    // ADD THIS - Simple fallback for FP32
    void simple_gemm_f32(
        const float* A, 
        const float* B,
        float* C,
        int M, int N, int K
    );
    
private:
    void* kernel_handle;  // Your compiled kernel library
};

} // namespace edgemind

#endif