#define _CRT_SECURE_NO_WARNINGS

#include "pack_loader.h"
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstring>

static std::string slurp(const std::string& path){
  FILE* f = std::fopen(path.c_str(), "rb");
  if(!f) throw std::runtime_error("open failed: " + path);
  std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  std::string s; s.resize((size_t)sz);
  if(sz>0 && std::fread(s.data(), 1, (size_t)sz, f) != (size_t)sz) { std::fclose(f); throw std::runtime_error("read failed: " + path); }
  std::fclose(f); return s;
}
static void readbin(const std::string& path, std::vector<uint8_t>& out){
  FILE* f = std::fopen(path.c_str(), "rb");
  if(!f) throw std::runtime_error("open failed: " + path);
  std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  out.resize((size_t)sz);
  if(sz>0 && std::fread(out.data(), 1, (size_t)sz, f) != (size_t)sz) { std::fclose(f); throw std::runtime_error("read failed: " + path); }
  std::fclose(f);
}

static int find_int(const std::string& js, const char* key){
  auto k = js.find(std::string("\"") + key + "\"");
  if(k==std::string::npos) throw std::runtime_error(std::string("key not found: ")+key);
  auto c = js.find_first_of("-0123456789", k);
  auto e = js.find_first_not_of("-0123456789", c);
  return std::stoi(js.substr(c, e-c));
}

PackedB load_q4edge_packed(const std::string& prefix){
  auto j = slurp(prefix + ".json");
  PackedB pb;
  pb.K = find_int(j, "K");
  pb.N = find_int(j, "N");
  pb.group = find_int(j, "group_size");
  pb.bytes_per_group = find_int(j, "bytes_per_group");
  int groups = (pb.K + pb.group - 1) / pb.group;

  std::vector<uint8_t> dbuf; readbin(prefix + ".bin", dbuf);
  std::vector<uint8_t> sbuf; readbin(prefix + ".scales.fp16.bin", sbuf);
  if((int)sbuf.size() != pb.N * groups * 2) throw std::runtime_error("scale size mismatch");
  if((int)dbuf.size() != pb.N * groups * pb.bytes_per_group) throw std::runtime_error("data size mismatch");

  pb.data.reset(new uint8_t[dbuf.size()]);
  pb.scales.reset(new uint16_t[sbuf.size()/2]);
  std::memcpy(pb.data.get(), dbuf.data(), dbuf.size());
  std::memcpy(pb.scales.get(), sbuf.data(), sbuf.size());
  return pb;
}
