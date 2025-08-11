// src/kernels/cpu/int4/model_interface.cpp
// FIXED VERSION - No BLAS, no undefined functions

#include "model_interface.h"
#include <cstring>
#include <thread>
#include <immintrin.h>

// Declare your existing kernel functions
extern "C" {
    // These should match what's in your qgemm_int4.cpp
    void qgemm_q8_mt(
        const float* A,
        const int8_t* B_q8, 
        const float* scales,
        const float* bias,  // Can be nullptr
        float* C,
        int M, int N, int K,
        int group_size,
        int num_threads
    );
}

namespace edgemind {

ModelKernel::ModelKernel() {
    // Nothing to initialize - kernels are just functions
}

Tensor ModelKernel::forward_linear(const Tensor& input, const LinearLayer& layer) {
    int M = input.dims[0] * input.dims[1];  // Batch * Seq
    int K = layer.in_features;
    int N = layer.out_features;
    
    // Allocate output
    Tensor output;
    output.dims[0] = input.dims[0];
    output.dims[1] = input.dims[1];
    output.dims[2] = N;
    output.numel = M * N;
    output.data_fp32 = new float[output.numel];
    
    // CALL YOUR 125 GFLOP/s KERNEL HERE!
    if (layer.weight.quantized) {
        // Use YOUR quantized kernel
        gemm_int8(
            input.data_fp32,
            layer.weight.data_int8,
            layer.weight.scales,
            output.data_fp32,
            M, N, K
        );
    } else {
        // Simple fallback GEMM (no BLAS needed)
        simple_gemm_f32(
            input.data_fp32,
            layer.weight.data_fp32,
            output.data_fp32,
            M, N, K
        );
    }
    
    // Add bias if present
    if (layer.bias.data_fp32) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                output.data_fp32[i * N + j] += layer.bias.data_fp32[j];
            }
        }
    }
    
    return output;
}

void ModelKernel::simple_gemm_f32(
    const float* A, 
    const float* B,
    float* C,
    int M, int N, int K
) {
    // Simple fallback for FP32 (not optimized, just works)
    // A is M x K, B is K x N, C is M x N
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

void ModelKernel::gemm_int8(
    const float* A, 
    const int8_t* B_quant,
    const float* scales,
    float* C,
    int M, int N, int K
) {
    // Call your existing Q8 kernel
    int num_threads = std::thread::hardware_concurrency();
    
    // Call the actual kernel from qgemm_int4.cpp
    qgemm_q8_mt(
        A,           // Input activations
        B_quant,     // Quantized weights
        scales,      // Scales
        nullptr,     // No bias in this path
        C,           // Output
        M, N, K,
        64,          // group_size
        num_threads
    );
}

} // namespace edgemind