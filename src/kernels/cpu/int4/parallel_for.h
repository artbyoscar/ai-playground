#pragma once
#include <thread>
#include <vector>
#include <algorithm>
#include <functional>

inline void parallel_for(int begin, int end, int num_threads, const std::function<void(int,int)>& fn) {
  num_threads = std::max(1, std::min(num_threads, end - begin));
  int total = end - begin;
  int chunk = (total + num_threads - 1) / num_threads;
  std::vector<std::thread> ts;
  ts.reserve(num_threads);
  for (int t=0; t<num_threads; ++t) {
    int s = begin + t*chunk;
    int e = std::min(end, s + chunk);
    if (s >= e) break;
    ts.emplace_back([=,&fn](){ fn(s, e); });
  }
  for (auto& th : ts) th.join();
}
