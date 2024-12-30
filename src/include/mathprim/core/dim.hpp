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

using dim_t = dim<max_supported_dim>;  ///< The default dimensionality type for general buffers.

template <>
struct dim<1> {
  explicit MATHPRIM_PRIMFUNC dim(index_t x) : x(x) {}

  MATHPRIM_COPY(dim, default);
  MATHPRIM_MOVE(dim, default);

  MATHPRIM_PRIMFUNC size_t numel() const noexcept { return internal::to_valid_size(x); }

  MATHPRIM_PRIMFUNC index_t ndim() const noexcept { return internal::is_valid_size(x) ? 1 : 0; }

  MATHPRIM_PRIMFUNC index_t size(index_t i) const {
    MATHPRIM_ASSERT(i == 0);
    return x;
  }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  index_t x;  ///< The x dimension.
};

template <>
struct dim<2> {
  MATHPRIM_PRIMFUNC dim(index_t x, index_t y) : x(x), y(y) {}

  MATHPRIM_COPY(dim, default);
  MATHPRIM_MOVE(dim, default);

  explicit MATHPRIM_PRIMFUNC dim(const dim<1> &d) : x(d.x), y(no_dim) {}

  // common functionalities.
  MATHPRIM_PRIMFUNC size_t numel() const noexcept {
    return internal::to_valid_size(x) * internal::to_valid_size(y);
  }

  MATHPRIM_PRIMFUNC index_t ndim() const noexcept {
    if (!internal::is_valid_size(x)) {
      return 0;
    } else if (!internal::is_valid_size(y)) {
      return 1;
    } else {
      return 2;
    }
  }

  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= 0 && i < 2);
    return i == 0 ? x : y;
  }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  // data
  index_t x;  ///< The x dimension.
  index_t y;  ///< The y dimension.
};

template <>
struct dim<3> {
  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z) : x(x), y(y), z(z) {}

  MATHPRIM_COPY(dim, default);
  MATHPRIM_MOVE(dim, default);

  // construct from lower dimensions
  explicit MATHPRIM_PRIMFUNC dim(const dim<1> &d) : x(d.x), y(no_dim), z(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<2> &d) : x(d.x), y(d.y), z(no_dim) {}

  // common functionalities.
  MATHPRIM_PRIMFUNC size_t numel() const noexcept {
    return internal::to_valid_size(x) * internal::to_valid_size(y) * internal::to_valid_size(z);
  }

  MATHPRIM_PRIMFUNC index_t ndim() const noexcept {
    if (!internal::is_valid_size(x)) {
      return 0;
    } else if (!internal::is_valid_size(y)) {
      return 1;
    } else if (!internal::is_valid_size(z)) {
      return 2;
    } else {
      return 3;
    }
  }

  MATHPRIM_PRIMFUNC index_t size(index_t i) const {
    MATHPRIM_ASSERT(i >= 0 && i < 3);
    return i == 0 ? x : (i == 1 ? y : z);
  }

  MATHPRIM_PRIMFUNC dim<2> xy() const { return dim<2>{x, y}; }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  index_t x;  ///< The x dimension.
  index_t y;  ///< The y dimension.
  index_t z;  ///< The z dimension.
};

template <>
struct dim<4> {
  explicit MATHPRIM_PRIMFUNC dim(index_t x) : x(x), y(no_dim), z(no_dim), w(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y) : x(x), y(y), z(no_dim), w(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z) : x(x), y(y), z(z), w(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z, index_t w) : x(x), y(y), z(z), w(w) {}

  MATHPRIM_COPY(dim, default);
  MATHPRIM_MOVE(dim, default);

  // common functionalities.
  MATHPRIM_PRIMFUNC size_t numel() const noexcept {
    return internal::to_valid_size(x) * internal::to_valid_size(y) * internal::to_valid_size(z)
           * internal::to_valid_size(w);
  }

  MATHPRIM_PRIMFUNC index_t ndim() const noexcept {
    if (!internal::is_valid_size(x)) {
      return 0;
    } else if (!internal::is_valid_size(y)) {
      return 1;
    } else if (!internal::is_valid_size(z)) {
      return 2;
    } else if (!internal::is_valid_size(w)) {
      return 3;
    } else {
      return 4;
    }
  }

  // construct from lower dimensions
  explicit MATHPRIM_PRIMFUNC dim(const dim<1> &d) : x(d.x), y(no_dim), z(no_dim), w(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<2> &d) : x(d.x), y(d.y), z(no_dim), w(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<3> &d) : x(d.x), y(d.y), z(d.z), w(no_dim) {}

  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= 0 && i < 4);
    return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
  }

  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept { return size(i); }

  MATHPRIM_PRIMFUNC dim<2> xy() const { return dim<2>{x, y}; }

  MATHPRIM_PRIMFUNC dim<3> xyz() const { return dim<3>{x, y, z}; }

  index_t x;  ///< The x dimension.
  index_t y;  ///< The y dimension.
  index_t z;  ///< The z dimension.
  index_t w;  ///< The w dimension.
};

// Deduction guides.
dim(index_t x) -> dim<1>;
dim(index_t x, index_t y) -> dim<2>;
dim(index_t x, index_t y, index_t z) -> dim<3>;
dim(index_t x, index_t y, index_t z, index_t w) -> dim<4>;

///////////////////////////////////////////////////////////////////////////////
/// general helper functions
///////////////////////////////////////////////////////////////////////////////

// Returns the total number of elements
template <index_t N>
size_t numel(const dim<N> &d) {
  return d.numel();
}

// Returns the number of valid dimensions.
template <index_t N>
index_t ndim(const dim<N> &d) {
  return d.ndim();
}

// Returns the size of the i-th dimension.
template <index_t N>
index_t size(const dim<N> &d, index_t i) {
  return d.size(i);
}

// Returns the size of the i-th dimension, compile time version.
template <index_t i, index_t N>
index_t size(const dim<N> &d) {
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

}  // namespace mathprim
