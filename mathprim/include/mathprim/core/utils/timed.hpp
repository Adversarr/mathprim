#pragma once
#include <chrono>
#include <map>

namespace mathprim::internal {

class global_timer {
public:
  struct entry {
    std::chrono::duration<int64_t, std::nano> total_time_;
    int64_t n_calls_;
  };

  std::map<std::string, entry> entries_;

  static global_timer& instance() {
    static global_timer inst;
    return inst;
  }

  struct auto_timer {
    explicit auto_timer(const std::string& name) : entry_(instance().entries_[name]) {
      start_ = std::chrono::high_resolution_clock::now();
    }

    ~auto_timer() {
      auto end = std::chrono::high_resolution_clock::now();
      entry_.total_time_ += end - start_;
      entry_.n_calls_++;
    }

    entry& entry_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  };

private:
  global_timer() = default;

  ~global_timer() {
    for (const auto& [name, entry] : entries_) {
      float in_ms = std::chrono::duration<float, std::milli>(entry.total_time_).count();
      float avg = in_ms / entry.n_calls_;
      printf("%s: %.2f ms (%ld calls), avg: %.2f ms\n", name.c_str(), in_ms, entry.n_calls_, avg);
    }
  }
};

#define MATHPRIM_TIMED(expr)                                      \
  {                                                               \
    ::mathprim::internal::global_timer::auto_timer _timer(#expr); \
    expr;                                                         \
  }

}  // namespace mathprim::internal