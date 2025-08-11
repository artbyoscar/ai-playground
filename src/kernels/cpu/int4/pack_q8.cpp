// pack_q8.cpp
#include "pack_q8.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Helper for FP16 conversion
static inline uint16_t fp16_from_fp32(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t t = mant >> (1 - exp + 13);
        if ((mant >> (1 - exp + 12)) & 1u) t += 1u;
        return (uint16_t)(sign | t);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7BFFu);
    }
    
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) half += 1u;
    return (uint16_t)half;
}

Q8PackResult q8_pack_column_f32(
    const float* col, 
    int K, 
    int group_size) 
{
    Q8PackResult result;
    const int groups = (K + group_size - 1) / group_size;
    
    result.data.reserve(K);
    result.scales.reserve(groups);
    
    for (int g = 0; g < groups; ++g) {
        int k_start = g * group_size;
        int k_end = std::min(k_start + group_size, K);
        int group_k = k_end - k_start;
        
        // Find max absolute value in group
        float max_abs = 0.0f;
        for (int k = k_start; k < k_end; ++k) {
            max_abs = std::max(max_abs, std::abs(col[k]));
        }
        
        // Compute scale (avoid division by zero)
        float scale = (max_abs > 1e-6f) ? (max_abs / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;
        
        // Quantize values
        for (int k = k_start; k < k_end; ++k) {
            int q = (int)std::round(col[k] * inv_scale);
            q = std::max(-127, std::min(127, q));  // Clamp to int8 range
            result.data.push_back((int8_t)q);
        }
        
        // Pad with zeros if needed
        for (int k = group_k; k < group_size; ++k) {
            result.data.push_back(0);
        }
        
        // Store scale as FP16
        result.scales.push_back(fp16_from_fp32(scale));
    }
    
    return result;
}

Q8PackResult q8_pack_rowmajor_f32(
    const float* B, 
    int rows, 
    int cols, 
    int group_size) 
{
    Q8PackResult result;
    const int groups = (rows + group_size - 1) / group_size;
    
    result.data.reserve(rows * cols);
    result.scales.reserve(cols * groups);
    
    // Pack column by column
    for (int c = 0; c < cols; ++c) {
        // Extract column
        std::vector<float> col(rows);
        for (int r = 0; r < rows; ++r) {
            col[r] = B[r * cols + c];
        }
        
        // Pack this column
        Q8PackResult col_pack = q8_pack_column_f32(col.data(), rows, group_size);
        
        // Append to results
        result.data.insert(result.data.end(), 
                          col_pack.data.begin(), 
                          col_pack.data.end());
        result.scales.insert(result.scales.end(), 
                            col_pack.scales.begin(), 
                            col_pack.scales.end());
    }
    
    return result;
}

float q8_compute_error(
    const float* original,
    const int8_t* quantized,
    const uint16_t* scales,
    int K,
    int group_size)
{
    const int groups = (K + group_size - 1) / group_size;
    float total_error = 0.0f;
    float total_magnitude = 0.0f;
    
    // Helper to convert FP16 to FP32
    auto fp16_to_fp32 = [](uint16_t h) -> float {
        uint32_t sign = (uint32_t)(h & 0x8000) << 16;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t bits;
        
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                int e = -1;
                do { mant <<= 1; ++e; } while ((mant & 0x400) == 0);
                mant &= 0x3FF;
                uint32_t exp_f = (127 - 15 - e);
                bits = sign | (exp_f << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000u | (mant << 13);
        } else {
            uint32_t exp_f = exp - 15 + 127;
            bits = sign | (exp_f << 23) | (mant << 13);
        }
        
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    };
    
    for (int g = 0; g < groups; ++g) {
        float scale = fp16_to_fp32(scales[g]);
        int k_start = g * group_size;
        int k_end = std::min(k_start + group_size, K);
        
        for (int k = k_start; k < k_end; ++k) {
            float orig = original[k];
            float dequant = (float)quantized[k] * scale;
            float error = orig - dequant;
            
            total_error += error * error;
            total_magnitude += orig * orig;
        }
    }
    
    return (total_magnitude > 0) ? std::sqrt(total_error / total_magnitude) : 0.0f;
}