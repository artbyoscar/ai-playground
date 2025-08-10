#pragma once
#include <cstdint>
#include <vector>

// === Compact Q4 (nibble) layout ===========================================
// - group_size is typically 64
// - data : packed 4-bit weights (2 weights per byte), row-major groups
// - scales: fp16 (uint16_t bits), one per group
struct Q4EdgePacked {
  uint32_t rows;
  uint32_t cols;
  uint32_t group_size;          // usually 64
  std::vector<uint8_t>  data;   // packed nibbles
  std::vector<uint16_t> scales; // per-group fp16 scale
};

// Pack row-major float32 -> Q4 nibble layout
Q4EdgePacked
q4edge_pack_rowmajor_f32(const float* W, uint32_t rows, uint32_t cols, uint32_t group = 64);


// === Expanded Q8 (speed path) layout =======================================
// Same grouping/scaling as Q4, but each quantized weight occupies one int8
// [-8..7]. This removes the nibble-unpack cost at runtime.
//
// - data : int8 per weight, 64 bytes per group
// - scales: fp16 (uint16_t bits), one per group
struct Q4EdgePackedQ8 {
  uint32_t rows;
  uint32_t cols;
  uint32_t group_size;           // usually 64
  std::vector<int8_t>  data;     // expanded int8 weights
  std::vector<uint16_t> scales;  // per-group fp16 scale
};

// Pack row-major float32 -> Q8 expanded layout (weights in int8, [-8..7])
Q4EdgePackedQ8
q4edge_pack_rowmajor_f32_q8(const float* W, uint32_t rows, uint32_t cols, uint32_t group = 64);
