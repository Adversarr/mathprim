#pragma once
#include "mathprim/core/dim.hpp"

namespace mathprim {

template <index_t N>
class dim_iterator final {
public:
  using value_type = dim<N>;

  MATHPRIM_PRIMFUNC dim_iterator(const dim<N>& start, const dim<N>& end)
      : current_(start), shape_(end) {}

  MATHPRIM_PRIMFUNC dim_iterator& operator++() {
    for (index_t i = N - 1; i > 0; --i) {
      if (shape_[i] == no_dim) {
        continue;
      }
      if (++current_[i] < shape_[i]) {
        return *this;
      }
      current_[i] = 0;
    }
    ++current_[0];
    return *this;
  }

  MATHPRIM_PRIMFUNC dim_iterator operator++(int) {
    dim_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  MATHPRIM_PRIMFUNC const value_type& operator*() const { return current_; }

  MATHPRIM_PRIMFUNC bool operator!=(const dim_iterator& other) const {
    return current_ != other.current_;
  }

  MATHPRIM_PRIMFUNC bool operator==(const dim_iterator& other) const {
    return current_ == other.current_;
  }

private:
  dim<N> current_;
  dim<N> shape_;
};

/// ADL begin
template <index_t N>
MATHPRIM_PRIMFUNC dim_iterator<N> begin(const dim<N>& shape) {
  return dim_iterator<N>(dim<N>(), shape);
}

/// ADL end
template <index_t N>
MATHPRIM_PRIMFUNC dim_iterator<N> end(const dim<N>& shape) {
  return dim_iterator<N>(dim<N>(shape.size(0)), shape);
}

}  // namespace mathprim
