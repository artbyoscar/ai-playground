// tests/test_q8_pack.cpp
#include "../pack_q8.h"
#include <vector>
#include <random>
#include <cstdio>

int main() {
    const int K = 2048, N = 256;
    const int group_size = 64;
    
    // Generate random matrix
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> B(K * N);
    for (auto& v : B) v = dist(rng);
    
    // Pack to Q8
    printf("Packing %dx%d matrix to Q8 (group_size=%d)...\n", K, N, group_size);
    auto packed = q8_pack_rowmajor_f32(B.data(), K, N, group_size);
    
    printf("Packed sizes:\n");
    printf("  Data: %zu bytes\n", packed.data.size());
    printf("  Scales: %zu bytes\n", packed.scales.size() * sizeof(uint16_t));
    
    // Compute error for first column
    std::vector<float> col0(K);
    for (int k = 0; k < K; ++k) col0[k] = B[k * N];
    
    float error = q8_compute_error(
        col0.data(),
        packed.data.data(),
        packed.scales.data(),
        K, group_size
    );
    
    printf("Relative error (column 0): %.4e\n", error);
    
    // Save to disk
    FILE* f = fopen("test_q8.bin", "wb");
    if (f) {
        fwrite(packed.data.data(), 1, packed.data.size(), f);
        fclose(f);
        printf("Saved Q8 data to test_q8.bin\n");
    }
    
    return 0;
}