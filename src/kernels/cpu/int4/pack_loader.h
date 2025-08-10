#pragma once
#include <cstdint>
#include <memory>
#include <string>
struct PackedB {
  int K=0, N=0, group=64, bytes_per_group=32;
  std::unique_ptr<uint8_t[]>  data;    // size = N * groups * bytes_per_group
  std::unique_ptr<uint16_t[]> scales;  // size = N * groups (fp16)
};
PackedB load_q4edge_packed(const std::string& prefix); // prefix without extension
