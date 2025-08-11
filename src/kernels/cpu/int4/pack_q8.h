// pack_q8.h
#pragma once
#include <vector>
#include <cstdint>

struct Q8PackResult {
    std::vector<int8_t> data;      // Quantized int8 values
    std::vector<uint16_t> scales;  // FP16 scales per group
};

// Pack a single column to Q8
Q8PackResult q8_pack_column_f32(
    const float* col, 
    int K, 
    int group_size);

// Pack row-major matrix to Q8 (column-wise)
Q8PackResult q8_pack_rowmajor_f32(
    const float* B, 
    int rows, 
    int cols, 
    int group_size);

// Compute relative error
float q8_compute_error(
    const float* original,
    const int8_t* quantized,
    const uint16_t* scales,
    int K,
    int group_size);