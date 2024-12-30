/**
 * @file
 * @brief Dim/Shape definition. We support up to N=4
 */

#pragma once

#include "defines.hpp"

namespace mathprim {

namespace internal {

MATHPRIM_PRIMFUNC bool is_valid_size(index_t size) {
  return size != no_dim;
}

MATHPRIM_PRIMFUNC index_t to_valid_size(index_t size) {
  return is_valid_size(size) ? size : 1;
}

}  // namespace internal

///////////////////////////////////////////////////////////////////////////////
/// Template Implementations for dim.
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Dimensionality template.
 */
template <index_t N>
struct dim {
  static_assert(0 < N && N <= max_supported_dim, "dim<N> only supports N in [0, 4].");
  MATHPRIM_PRIMFUNC size_t numel() const noexcept;
  MATHPRIM_PRIMFUNC index_t ndim() const noexcept;
  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept;
  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept;
};

template <>
struct dim<1> {
  explicit MATHPRIM_PRIMFUNC dim(index_t x) : x_(x) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<2> &d);
  explicit MATHPRIM_PRIMFUNC dim(const dim<3> &d);
  explicit MATHPRIM_PRIMFUNC dim(const dim<4> &d);

  MATHPRIM_COPY(dim, default);
  MATHPRIM_MOVE(dim, default);

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

  MATHPRIM_COPY(dim, default);
  MATHPRIM_MOVE(dim, default);

  explicit MATHPRIM_PRIMFUNC dim(const dim<1> &d) : x_(d.x_), y_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<3> &d);
  explicit MATHPRIM_PRIMFUNC dim(const dim<4> &d);

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

  MATHPRIM_COPY(dim, default);
  MATHPRIM_MOVE(dim, default);

  // construct from lower dimensions
  explicit MATHPRIM_PRIMFUNC dim(const dim<1> &d) : x_(d.x_), y_(no_dim), z_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<2> &d) : x_(d.x_), y_(d.y_), z_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<4> &d);

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

  MATHPRIM_PRIMFUNC dim<2> xy() const { return dim<2>{x_, y_}; }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  index_t x_;  ///< The x dimension.
  index_t y_;  ///< The y dimension.
  index_t z_;  ///< The z dimension.
};

template <>
struct dim<4> {
  MATHPRIM_PRIMFUNC dim() : x_(no_dim), y_(no_dim), z_(no_dim), w_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(index_t x) : x_(x), y_(no_dim), z_(no_dim), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y) : x_(x), y_(y), z_(no_dim), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z) : x_(x), y_(y), z_(z), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z, index_t w) : x_(x), y_(y), z_(z), w_(w) {}

  MATHPRIM_COPY(dim, default);
  MATHPRIM_MOVE(dim, default);

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
  explicit MATHPRIM_PRIMFUNC dim(const dim<1> &d) : x_(d.x_), y_(no_dim), z_(no_dim), w_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<2> &d) : x_(d.x_), y_(d.y_), z_(no_dim), w_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<3> &d) : x_(d.x_), y_(d.y_), z_(d.z_), w_(no_dim) {}

  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= 0 && i < 4);
    return i == 0 ? x_ : (i == 1 ? y_ : (i == 2 ? z_ : w_));
  }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  MATHPRIM_PRIMFUNC dim<2> xy() const { return dim<2>{x_, y_}; }

  MATHPRIM_PRIMFUNC dim<3> xyz() const { return dim<3>{x_, y_, z_}; }

  index_t x_;  ///< The x dimension.
  index_t y_;  ///< The y dimension.
  index_t z_;  ///< The z dimension.
  index_t w_;  ///< The w dimension.
};

// Implements the constructors.
MATHPRIM_PRIMFUNC dim<1>::dim(const dim<2> &d) : x_(d.x_) {
  MATHPRIM_ASSERT(d.ndim() <= 1);
}

MATHPRIM_PRIMFUNC dim<1>::dim(const dim<3> &d) : x_(d.x_) {
  MATHPRIM_ASSERT(d.ndim() <= 1);
}

MATHPRIM_PRIMFUNC dim<1>::dim(const dim<4> &d) : x_(d.x_) {
  MATHPRIM_ASSERT(d.ndim() <= 1);
}

MATHPRIM_PRIMFUNC dim<2>::dim(const dim<3> &d) : x_(d.x_), y_(d.y_) {
  MATHPRIM_ASSERT(d.ndim() <= 2);
}

MATHPRIM_PRIMFUNC dim<2>::dim(const dim<4> &d) : x_(d.x_), y_(d.y_) {
  MATHPRIM_ASSERT(d.ndim() <= 2);
}

MATHPRIM_PRIMFUNC dim<3>::dim(const dim<4> &d) : x_(d.x_), y_(d.y_), z_(d.z_) {
  MATHPRIM_ASSERT(d.ndim() <= 3);
}

// Deduction guides.
dim(index_t x) -> dim<1>;
dim(index_t x, index_t y) -> dim<2>;
dim(index_t x, index_t y, index_t z) -> dim<3>;
dim(index_t x, index_t y, index_t z, index_t w) -> dim<4>;

///////////////////////////////////////////////////////////////////////////////
/// Comparison operators
///////////////////////////////////////////////////////////////////////////////

MATHPRIM_PRIMFUNC bool operator==(const dim<1> &lhs, const dim<1> &rhs) {
  return lhs.x_ == rhs.x_;
}

MATHPRIM_PRIMFUNC bool operator==(const dim<2> &lhs, const dim<2> &rhs) {
  return lhs.x_ == rhs.x_ && lhs.y_ == rhs.y_;
}

MATHPRIM_PRIMFUNC bool operator==(const dim<3> &lhs, const dim<3> &rhs) {
  return lhs.x_ == rhs.x_ && lhs.y_ == rhs.y_ && lhs.z_ == rhs.z_;
}

MATHPRIM_PRIMFUNC bool operator==(const dim<4> &lhs, const dim<4> &rhs) {
  return lhs.x_ == rhs.x_ && lhs.y_ == rhs.y_ && lhs.z_ == rhs.z_ && lhs.w_ == rhs.w_;
}

template <index_t N>
MATHPRIM_PRIMFUNC bool operator!=(const dim<N> &lhs, const dim<N> &rhs) {
  return !(lhs == rhs);
}

///////////////////////////////////////////////////////////////////////////////
/// general helper functions
///////////////////////////////////////////////////////////////////////////////

// Returns the total number of elements
template <index_t N>
MATHPRIM_PRIMFUNC size_t numel(const dim<N> &d) {
  return d.numel();
}

// Returns the number of valid dimensions.
template <index_t N>
MATHPRIM_PRIMFUNC index_t ndim(const dim<N> &d) {
  return d.ndim();
}

// Returns the size of the i-th dimension.
template <index_t N>
MATHPRIM_PRIMFUNC index_t size(const dim<N> &d, index_t i) {
  return d.size(i);
}

// Returns the size of the i-th dimension, compile time version.
template <index_t i, index_t N>
MATHPRIM_PRIMFUNC index_t size(const dim<N> &d) {
  static_assert(i < N, "Index out of bounds.");
  return d.size(i);  // NOTE: most compilers will optimize this?
}

///////////////////////////////////////////////////////////////////////////////
/// Helpers for strides.
///////////////////////////////////////////////////////////////////////////////
// Return a row-majored stride, similar to numpy.
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

MATHPRIM_PRIMFUNC index_t sub2ind(const dim<1> &stride, const dim<1> &sub) {
  return sub.x_ * stride.x_;
}

MATHPRIM_PRIMFUNC index_t sub2ind(const dim<2> &stride, const dim<2> &sub) {
  auto [x, y] = sub;
  auto [sx, sy] = stride;
  return x * sx + y * sy;
}

MATHPRIM_PRIMFUNC index_t sub2ind(const dim<3> &stride, const dim<3> &sub) {
  auto [x, y, z] = sub;
  auto [sx, sy, sz] = stride;
  return x * sx + y * sy + z * sz;
}

MATHPRIM_PRIMFUNC index_t sub2ind(const dim<4> &stride, const dim<4> &sub) {
  auto [x, y, z, w] = sub;
  auto [sx, sy, sz, sw] = stride;
  return x * sx + y * sy + z * sz + w * sw;
}

MATHPRIM_PRIMFUNC void check_in_bounds(const dim<1> &shape, const dim<1> &sub) {
  MATHPRIM_ASSERT(sub.x_ >= 0 && sub.x_ < shape.x_);
}

MATHPRIM_PRIMFUNC void check_in_bounds(const dim<2> &shape, const dim<2> &sub) {
  MATHPRIM_ASSERT(sub.x_ >= 0 && sub.x_ < shape.x_);
  MATHPRIM_ASSERT(sub.y_ >= 0 && sub.y_ < shape.y_);
}

MATHPRIM_PRIMFUNC void check_in_bounds(const dim<3> &shape, const dim<3> &sub) {
  MATHPRIM_ASSERT(sub.x_ >= 0 && sub.x_ < shape.x_);
  MATHPRIM_ASSERT(sub.y_ >= 0 && sub.y_ < shape.y_);
  MATHPRIM_ASSERT(sub.z_ >= 0 && sub.z_ < shape.z_);
}

MATHPRIM_PRIMFUNC void check_in_bounds(const dim<4> &shape, const dim<4> &sub) {
  MATHPRIM_ASSERT(sub.x_ >= 0 && sub.x_ < shape.x_);
  MATHPRIM_ASSERT(sub.y_ >= 0 && sub.y_ < shape.y_);
  MATHPRIM_ASSERT(sub.z_ >= 0 && sub.z_ < shape.z_);
  MATHPRIM_ASSERT(sub.w_ >= 0 && sub.w_ < shape.w_);
}

}  // namespace mathprim
