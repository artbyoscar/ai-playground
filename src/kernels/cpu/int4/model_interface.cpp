// src/kernels/cpu/int4/model_interface.cpp
#include "model_interface.h"
#include "qgemm_int4.h"  // Your existing kernel header
#include <cstring>
#include <thread>

namespace edgemind {

ModelKernel::ModelKernel() {
    // Initialize your kernel system
    init_int4_kernels();
}

Tensor ModelKernel::forward_linear(const Tensor& input, const LinearLayer& layer) {
    // This is THE CRITICAL FUNCTION - bridges models to your kernels
    
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
        // Fallback to regular GEMM
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                   M, N, K, 1.0f,
                   input.data_fp32, K,
                   layer.weight.data_fp32, K,
                   0.0f, output.data_fp32, N);
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

void ModelKernel::gemm_int8(
    const float* A, 
    const int8_t* B_quant,
    const float* scales,
    float* C,
    int M, int N, int K
) {
    // Call your existing kernel
    // This is where your 125 GFLOP/s code runs!
    int num_threads = std::thread::hardware_concurrency();
    
    qgemm_int8_mt(
        A, B_quant, scales,
        C, M, N, K,
        64,  // group_size
        num_threads
    );
}

} // namespace edgemind