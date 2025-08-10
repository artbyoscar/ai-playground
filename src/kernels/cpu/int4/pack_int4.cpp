#include "pack_int4.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstring>

// minimal fp16-from-fp32; swap for precise later
static inline uint16_t fp16_from_fp32(float f) {
  uint32_t x; std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;

  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    uint32_t t = mant >> (1 - exp + 13);
    if ((mant >> (1 - exp + 12)) & 1u) t += 1u; // round-to-nearest-even
    return (uint16_t)(sign | t);
  } else if (exp >= 31) {
    return (uint16_t)(sign | 0x7BFFu); // clamp to 65504
  } else {
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) half += 1u; // round-to-nearest-even
    return (uint16_t)half;
  }
}

// ========================= Q4 nibble packer ================================

Q4EdgePacked
q4edge_pack_rowmajor_f32(const float* W, uint32_t rows, uint32_t cols, uint32_t group) {
  Q4EdgePacked out{rows, cols, group};
  const uint32_t groups_per_row = (cols + group - 1) / group;
  out.scales.reserve(size_t(rows) * groups_per_row);
  out.data.reserve(size_t(rows) * groups_per_row * (group/2));

  for (uint32_t r=0; r<rows; ++r) {
    const float* row = W + size_t(r)*cols;

    for (uint32_t g=0; g<cols; g+=group) {
      const uint32_t gend = std::min(g+group, cols);

      // per-group scale
      float maxabs = 1e-8f;
      for (uint32_t c=g; c<gend; ++c) maxabs = std::max(maxabs, std::fabs(row[c]));
      float s = maxabs / 7.0f;
      if (s < 1e-8f) s = 1e-8f;
      out.scales.push_back(fp16_from_fp32(s));

      // two-at-a-time pack (hi nibble = q0, lo nibble = q1)
      for (uint32_t c=g; c<g+group; c+=2) {
        int q0 = 0, q1 = 0;
        if (c   < cols) q0 = (int)std::lround(row[c]   / s);
        if (c+1 < cols) q1 = (int)std::lround(row[c+1] / s);
        q0 = std::clamp(q0, -8, 7);
        q1 = std::clamp(q1, -8, 7);
        const uint8_t b = uint8_t(((q0+8)&0xF) << 4 | ((q1+8)&0xF));
        out.data.push_back(b);
      }
    }
  }
  return out;
}

// ========================= Q8 expanded packer ==============================

Q4EdgePackedQ8
q4edge_pack_rowmajor_f32_q8(const float* W, uint32_t rows, uint32_t cols, uint32_t group) {
  Q4EdgePackedQ8 out{rows, cols, group};
  const uint32_t groups_per_row = (cols + group - 1) / group;
  out.scales.reserve(size_t(rows) * groups_per_row);
  out.data.reserve(size_t(rows) * groups_per_row * group);

  for (uint32_t r=0; r<rows; ++r) {
    const float* row = W + size_t(r)*cols;

    for (uint32_t g=0; g<cols; g+=group) {
      const uint32_t gend = std::min(g+group, cols);

      // per-group scale
      float maxabs = 1e-8f;
      for (uint32_t c=g; c<gend; ++c) maxabs = std::max(maxabs, std::fabs(row[c]));
      float s = maxabs / 7.0f;
      if (s < 1e-8f) s = 1e-8f;
      out.scales.push_back(fp16_from_fp32(s));

      // one byte per element (int8), zero-pad to full group
      for (uint32_t c=g; c<gend; ++c) {
        int q = (int)std::nearbyint(row[c] / s);
        q = std::clamp(q, -8, 7);
        out.data.push_back((int8_t)q);
      }
      for (uint32_t c=gend; c<g+group; ++c) {
        out.data.push_back((int8_t)0);
      }
    }
  }
  return out;
}

