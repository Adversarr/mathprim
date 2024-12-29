/**
 * @file
 * @brief Dim/Shape definition. We support up to N=4
 */

#pragma once

#include "defines.hpp"

namespace mathprim {

// Indicates this dimension does not exist logically.
constexpr index_t no_dim = 0;

// Indicates this dimension does not change under some operation.
constexpr index_t keep_dim = -1;

namespace internal {

MATHPRIM_PRIMFUNC bool is_valid_size(index_t size) {
  return size != no_dim;
}

MATHPRIM_PRIMFUNC index_t to_valid_size(index_t size) {
  return is_valid_size(size) ? size : 1;
}

}  // namespace internal

/**
 * @brief Dimensionality template.
 */
template <index_t N>
struct dim {
  static_assert(0 < N && N <= 4, "dim<N> only supports N in [0, 4].");
  MATHPRIM_PRIMFUNC size_t numel() const noexcept;
  MATHPRIM_PRIMFUNC index_t ndim() const noexcept;
  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept;
  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept;
};

using dim1 = dim<1>;   ///< 1D dimensionality type.
using dim2 = dim<2>;   ///< 2D dimensionality type.
using dim3 = dim<3>;   ///< 3D dimensionality type.
using dim4 = dim<4>;   ///< 4D dimensionality type.
using dim_t = dim<4>;  ///< The default dimensionality type for general buffers.

template <>
struct dim<1> {
  MATHPRIM_PRIMFUNC dim(index_t x) : x_(x) {}

  MATHPRIM_PRIMCOPY(dim, default);
  MATHPRIM_PRIMMOVE(dim, default);

  MATHPRIM_PRIMFUNC size_t numel() const noexcept { return internal::to_valid_size(x_); }

  MATHPRIM_PRIMFUNC index_t ndim() const noexcept { return internal::is_valid_size(x_) ? 1 : 0; }

  MATHPRIM_PRIMFUNC index_t size(index_t i) const {
    MATHPRIM_ASSERT(i == 0);
    return x_;
  }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  index_t x_;  ///< The x dimension.
};

template <>
struct dim<2> {
  MATHPRIM_PRIMFUNC dim(index_t x, index_t y) : x_(x), y_(y) {}

  MATHPRIM_PRIMCOPY(dim, default);
  MATHPRIM_PRIMMOVE(dim, default);

  MATHPRIM_PRIMFUNC dim(const dim1 &d) : x_(d.x_), y_(no_dim) {}

  // common functionalities.
  MATHPRIM_PRIMFUNC size_t numel() const noexcept {
    return internal::to_valid_size(x_) * internal::to_valid_size(y_);
  }

  MATHPRIM_PRIMFUNC index_t ndim() const noexcept {
    if (!internal::is_valid_size(x_)) {
      return 0;
    } else if (!internal::is_valid_size(y_)) {
      return 1;
    } else {
      return 2;
    }
  }

  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= 0 && i < 2);
    return i == 0 ? x_ : y_;
  }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  // data
  index_t x_;  ///< The x dimension.
  index_t y_;  ///< The y dimension.
};

template <>
struct dim<3> {
  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z) : x_(x), y_(y), z_(z) {}

  MATHPRIM_PRIMCOPY(dim, default);
  MATHPRIM_PRIMMOVE(dim, default);

  // construct from lower dimensions
  MATHPRIM_PRIMFUNC dim(const dim1 &d) : x_(d.x_), y_(no_dim), z_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(const dim2 &d) : x_(d.x_), y_(d.y_), z_(no_dim) {}

  // common functionalities.
  MATHPRIM_PRIMFUNC size_t numel() const noexcept {
    return internal::to_valid_size(x_) * internal::to_valid_size(y_) * internal::to_valid_size(z_);
  }

  MATHPRIM_PRIMFUNC index_t ndim() const noexcept {
    if (!internal::is_valid_size(x_)) {
      return 0;
    } else if (!internal::is_valid_size(y_)) {
      return 1;
    } else if (!internal::is_valid_size(z_)) {
      return 2;
    } else {
      return 3;
    }
  }

  MATHPRIM_PRIMFUNC index_t size(index_t i) const {
    MATHPRIM_ASSERT(i >= 0 && i < 3);
    return i == 0 ? x_ : (i == 1 ? y_ : z_);
  }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  index_t x_;  ///< The x dimension.
  index_t y_;  ///< The y dimension.
  index_t z_;  ///< The z dimension.
};

template <>
struct dim<4> {
  MATHPRIM_PRIMFUNC dim(index_t x) : x_(x), y_(no_dim), z_(no_dim), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y) : x_(x), y_(y), z_(no_dim), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z) : x_(x), y_(y), z_(z), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z, index_t w) : x_(x), y_(y), z_(z), w_(w) {}

  MATHPRIM_PRIMCOPY(dim, default);
  MATHPRIM_PRIMMOVE(dim, default);

  // common functionalities.
  MATHPRIM_PRIMFUNC size_t numel() const noexcept {
    return internal::to_valid_size(x_) * internal::to_valid_size(y_) * internal::to_valid_size(z_)
           * internal::to_valid_size(w_);
  }

  MATHPRIM_PRIMFUNC index_t ndim() const noexcept {
    if (!internal::is_valid_size(x_)) {
      return 0;
    } else if (!internal::is_valid_size(y_)) {
      return 1;
    } else if (!internal::is_valid_size(z_)) {
      return 2;
    } else if (!internal::is_valid_size(w_)) {
      return 3;
    } else {
      return 4;
    }
  }

  // construct from lower dimensions
  MATHPRIM_PRIMFUNC dim(const dim1 &d) : x_(d.x_), y_(no_dim), z_(no_dim), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(const dim2 &d) : x_(d.x_), y_(d.y_), z_(no_dim), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(const dim3 &d) : x_(d.x_), y_(d.y_), z_(d.z_), w_(no_dim) {}

  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= 0 && i < 4);
    return i == 0 ? x_ : (i == 1 ? y_ : (i == 2 ? z_ : w_));
  }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  index_t x_;  ///< The x dimension.
  index_t y_;  ///< The y dimension.
  index_t z_;  ///< The z dimension.
  index_t w_;  ///< The w dimension.
};

// Deduction guides.
dim(index_t x) -> dim<1>;
dim(index_t x, index_t y) -> dim<2>;
dim(index_t x, index_t y, index_t z) -> dim<3>;
dim(index_t x, index_t y, index_t z, index_t w) -> dim<4>;

// Helper functions to get the size of a dimension.
template <index_t N>
size_t numel(const dim<N> &d) {
  return d.numel();
}

template <index_t N>
index_t ndim(const dim<N> &d) {
  return d.ndim();
}

template <index_t N>
index_t size(const dim<N> &d, index_t i) {
  return d.size(i);
}

template <index_t i, index_t N>
index_t size(const dim<N> &d) {
  static_assert(i < N, "Index out of bounds.");
  return d.size(i);  // NOTE: most compilers will optimize this?
}

// Helpers for strides.
inline dim_t make_default_stride(const dim_t &shape) {
  index_t ndim = mathprim::ndim(shape);

  if (ndim == 0) {
    return dim_t{no_dim, no_dim, no_dim, no_dim};
  } else if (ndim == 1) {
    return dim_t{1, no_dim, no_dim, no_dim};
  } else if (ndim == 2) {
    return dim_t{shape[1], 1, no_dim, no_dim};
  } else if (ndim == 3) {
    return dim_t{shape[1] * shape[2], shape[2], 1, no_dim};
  } else if (ndim == 4) {
    return dim_t{shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1};
  }
  MATHPRIM_UNREACHABLE();
}

}  // namespace mathprim
