/**
 * @file
 * @brief Dim/Shape definition. We support up to N=4
 */

#pragma once

#include "defines.hpp"

namespace mathprim {

struct index_pair {
  index_t x_;
  index_t y_;
};

namespace internal {

constexpr MATHPRIM_PRIMFUNC bool is_valid_size(index_t size) {
  return size != no_dim;
}

constexpr MATHPRIM_PRIMFUNC index_t to_valid_index(index_t size) {
  return is_valid_size(size) ? size : 1;
}

constexpr MATHPRIM_PRIMFUNC index_pair div(index_t x, index_t y) {
  return index_pair{x / y, x % y};
}

} // namespace internal

///////////////////////////////////////////////////////////////////////////////
/// Template Implementations for dim.
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Dimensionality template.
 */
template <index_t N> struct dim {
  static_assert(0 < N && N <= max_ndim, "dim<N> only supports N in [0, 4].");
  MATHPRIM_PRIMFUNC index_t numel() const noexcept;
  MATHPRIM_PRIMFUNC index_t ndim() const noexcept;
  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept;
  MATHPRIM_PRIMFUNC index_t operator[](index_t i) const noexcept;
};

template <> struct dim<1> {
  MATHPRIM_PRIMFUNC dim() : x_(no_dim) {}

  constexpr explicit MATHPRIM_PRIMFUNC dim(index_t x) : x_(x) {}

  constexpr explicit MATHPRIM_PRIMFUNC dim(const dim<2> &d) noexcept;
  constexpr explicit MATHPRIM_PRIMFUNC dim(const dim<3> &d) noexcept;
  constexpr explicit MATHPRIM_PRIMFUNC dim(const dim<4> &d) noexcept;

  MATHPRIM_INTERNAL_COPY(dim, default);
  MATHPRIM_INTERNAL_MOVE(dim, default);

  constexpr MATHPRIM_PRIMFUNC index_t numel() const noexcept {
    return internal::to_valid_index(x_);
  }

  constexpr MATHPRIM_PRIMFUNC index_t ndim() const noexcept {
    return internal::is_valid_size(x_) ? 1 : 0;
  }

  constexpr MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept {
    MATHPRIM_ASSERT(i == 0);
    MATHPRIM_INTERNAL_MAYBE_UNUSED(i);
    return x_;
  }

  constexpr MATHPRIM_PRIMFUNC index_t &operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(i == 0);
    MATHPRIM_INTERNAL_MAYBE_UNUSED(i);
    return x_;
  }

  constexpr MATHPRIM_PRIMFUNC const index_t &
  operator[](index_t i) const noexcept { // NOLINT: unused
    MATHPRIM_ASSERT(i == 0);
    return x_;
  }

  constexpr explicit operator index_t() const noexcept { return x_; }

  index_t x_; ///< The x dimension.
};

template <> struct dim<2> {
  MATHPRIM_PRIMFUNC dim() noexcept : x_(no_dim), y_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(index_t x) noexcept : x_(x), y_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y) noexcept : x_(x), y_(y) {}

  MATHPRIM_INTERNAL_COPY(dim, default);
  MATHPRIM_INTERNAL_MOVE(dim, default);

  explicit MATHPRIM_PRIMFUNC dim(const dim<1> &d) noexcept
      : x_(d.x_), y_(no_dim) {}
  explicit MATHPRIM_PRIMFUNC dim(const dim<3> &d) noexcept;
  explicit MATHPRIM_PRIMFUNC dim(const dim<4> &d) noexcept;

  // common functionalities.
  MATHPRIM_PRIMFUNC index_t numel() const noexcept {
    return internal::to_valid_index(x_) * internal::to_valid_index(y_);
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
    MATHPRIM_ASSERT(i >= -1 && i < 2);
    return i == 0 ? x_ : y_;
  }

  MATHPRIM_PRIMFUNC index_t &operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(i >= -1 && i < 2);
    return i == 0 ? x_ : y_;
  }

  MATHPRIM_PRIMFUNC const index_t &operator[](index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= -1 && i < 2);
    return i == 0 ? x_ : y_;
  }

  // data
  index_t x_; ///< The x dimension.
  index_t y_; ///< The y dimension.
};

template <> struct dim<3> {
  MATHPRIM_PRIMFUNC dim() noexcept : x_(no_dim), y_(no_dim), z_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(index_t x) noexcept
      : x_(x), y_(no_dim), z_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y) noexcept
      : x_(x), y_(y), z_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z) noexcept
      : x_(x), y_(y), z_(z) {}

  MATHPRIM_INTERNAL_COPY(dim, default);
  MATHPRIM_INTERNAL_MOVE(dim, default);

  // construct from lower dimensions
  explicit MATHPRIM_PRIMFUNC dim(const dim<1> &d) noexcept
      : x_(d.x_), y_(no_dim), z_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<2> &d) noexcept
      : x_(d.x_), y_(d.y_), z_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<4> &d) noexcept;

  // common functionalities.
  MATHPRIM_PRIMFUNC index_t numel() const noexcept {
    return internal::to_valid_index(x_) * internal::to_valid_index(y_) *
           internal::to_valid_index(z_);
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
    MATHPRIM_ASSERT(i >= -2 && i < 3);
    if (i == 0) {
      return x_;
    } else if (i == 1 || i == -2) {
      return y_;
    } else if (i == 2 || i == -1) {
      return z_;
    }
    MATHPRIM_UNREACHABLE();
  }

  MATHPRIM_PRIMFUNC dim<2> xy() const { return dim<2>{x_, y_}; }

  MATHPRIM_PRIMFUNC index_t &operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(i >= -2 && i < 3);
    if (i == 0) {
      return x_;
    } else if (i == 1 || i == -2) {
      return y_;
    } else if (i == 2 || i == -1) {
      return z_;
    }
    MATHPRIM_UNREACHABLE();
  }

  MATHPRIM_PRIMFUNC const index_t &operator[](index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= -2 && i < 3);
    if (i == 0) {
      return x_;
    } else if (i == 1 || i == -2) {
      return y_;
    } else if (i == 2 || i == -1) {
      return z_;
    }
    MATHPRIM_UNREACHABLE();
  }

  index_t x_; ///< The x dimension.
  index_t y_; ///< The y dimension.
  index_t z_; ///< The z dimension.
};

template <> struct dim<4> {
  MATHPRIM_PRIMFUNC dim() : x_(no_dim), y_(no_dim), z_(no_dim), w_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(index_t x)
      : x_(x), y_(no_dim), z_(no_dim), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y)
      : x_(x), y_(y), z_(no_dim), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z)
      : x_(x), y_(y), z_(z), w_(no_dim) {}

  MATHPRIM_PRIMFUNC dim(index_t x, index_t y, index_t z, index_t w)
      : x_(x), y_(y), z_(z), w_(w) {}

  MATHPRIM_INTERNAL_COPY(dim, default);
  MATHPRIM_INTERNAL_MOVE(dim, default);

  // common functionalities.
  MATHPRIM_PRIMFUNC index_t numel() const noexcept {
    return internal::to_valid_index(x_) * internal::to_valid_index(y_) *
           internal::to_valid_index(z_) * internal::to_valid_index(w_);
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
  explicit MATHPRIM_PRIMFUNC dim(const dim<1> &d)
      : x_(d.x_), y_(no_dim), z_(no_dim), w_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<2> &d)
      : x_(d.x_), y_(d.y_), z_(no_dim), w_(no_dim) {}

  explicit MATHPRIM_PRIMFUNC dim(const dim<3> &d)
      : x_(d.x_), y_(d.y_), z_(d.z_), w_(no_dim) {}

  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= -3 && i < 4);
    if (i == 0) {
      return x_;
    } else if (i == 1 || i == -3) {
      return y_;
    } else if (i == 2 || i == -2) {
      return z_;
    } else if (i == 3 || i == -1) {
      return w_;
    }
    MATHPRIM_UNREACHABLE();
  }

  MATHPRIM_PRIMFUNC index_t &operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(i >= -3 && i < 4);
    if (i == 0) {
      return x_;
    } else if (i == 1 || i == -3) {
      return y_;
    } else if (i == 2 || i == -2) {
      return z_;
    } else if (i == 3 || i == -1) {
      return w_;
    }
    MATHPRIM_UNREACHABLE();
  }

  MATHPRIM_PRIMFUNC const index_t &operator[](index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= -3 && i < 4);
    if (i == 0) {
      return x_;
    } else if (i == 1 || i == -3) {
      return y_;
    } else if (i == 2 || i == -2) {
      return z_;
    } else if (i == 3 || i == -1) {
      return w_;
    }
    MATHPRIM_UNREACHABLE();
  }

  MATHPRIM_PRIMFUNC dim<2> xy() const { return dim<2>{x_, y_}; }

  MATHPRIM_PRIMFUNC dim<3> xyz() const { return dim<3>{x_, y_, z_}; }

  index_t x_; ///< The x dimension.
  index_t y_; ///< The y dimension.
  index_t z_; ///< The z dimension.
  index_t w_; ///< The w dimension.
};

// Implements the constructors.
constexpr MATHPRIM_PRIMFUNC dim<1>::dim(const dim<2> &d) noexcept : x_(d.x_) {
  MATHPRIM_ASSERT(d.ndim() <= 1);
}

constexpr MATHPRIM_PRIMFUNC dim<1>::dim(const dim<3> &d) noexcept : x_(d.x_) {
  MATHPRIM_ASSERT(d.ndim() <= 1);
}

constexpr MATHPRIM_PRIMFUNC dim<1>::dim(const dim<4> &d) noexcept : x_(d.x_) {
  MATHPRIM_ASSERT(d.ndim() <= 1);
}

MATHPRIM_PRIMFUNC dim<2>::dim(const dim<3> &d) noexcept : x_(d.x_), y_(d.y_) {
  MATHPRIM_ASSERT(d.ndim() <= 2);
}

MATHPRIM_PRIMFUNC dim<2>::dim(const dim<4> &d) noexcept : x_(d.x_), y_(d.y_) {
  MATHPRIM_ASSERT(d.ndim() <= 2);
}

MATHPRIM_PRIMFUNC dim<3>::dim(const dim<4> &d) noexcept
    : x_(d.x_), y_(d.y_), z_(d.z_) {
  MATHPRIM_ASSERT(d.ndim() <= 3);
}

// Deduction guides.
dim(index_t x) -> dim<1>;
dim(index_t x, index_t y) -> dim<2>;
dim(index_t x, index_t y, index_t z) -> dim<3>;
dim(index_t x, index_t y, index_t z, index_t w) -> dim<4>;

template <typename... Args>
MATHPRIM_PRIMFUNC dim<sizeof...(Args)> make_dim(Args &&...args) {
  return dim<sizeof...(Args)>{std::forward<Args>(args)...};
}

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
  return lhs.x_ == rhs.x_ && lhs.y_ == rhs.y_ && lhs.z_ == rhs.z_ &&
         lhs.w_ == rhs.w_;
}

template <index_t N>
MATHPRIM_PRIMFUNC bool operator!=(const dim<N> &lhs, const dim<N> &rhs) {
  return !(lhs == rhs);
}

///////////////////////////////////////////////////////////////////////////////
/// general helper functions
///////////////////////////////////////////////////////////////////////////////

// Returns the total number of elements
template <index_t N> MATHPRIM_PRIMFUNC index_t numel(const dim<N> &d) {
  return d.numel();
}

// Returns the number of valid dimensions.
template <index_t N> MATHPRIM_PRIMFUNC index_t ndim(const dim<N> &d) {
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
  return d.size(i); // NOTE: most compilers will optimize this?
}

///////////////////////////////////////////////////////////////////////////////
/// Helpers for strides.
///////////////////////////////////////////////////////////////////////////////
MATHPRIM_PRIMFUNC dim<1> make_default_stride(const dim<1> & /*shape*/) {
  return dim<1>{1};
}

MATHPRIM_PRIMFUNC dim<2> make_default_stride(const dim<2> &shape) {
  if (shape.y_ == no_dim) {
    return dim<2>{1, no_dim};
  } else {
    return dim<2>{shape.y_, 1};
  }
}

MATHPRIM_PRIMFUNC dim<3> make_default_stride(const dim<3> &shape) {
  if (shape.z_ == no_dim) {
    return dim<3>{1, no_dim, no_dim};
  } else if (shape.y_ == no_dim) {
    return dim<3>{shape.z_, 1, no_dim};
  } else {
    return dim<3>{shape.y_ * shape.z_, shape.z_, 1};
  }
}

// Return a row-majored stride, similar to numpy.
MATHPRIM_PRIMFUNC dim_t make_default_stride(const dim_t &shape) {
  index_t ndim = ::mathprim::ndim(shape);

  if (ndim == 0) {
    return dim_t{no_dim, no_dim, no_dim, no_dim};
  } else if (ndim == 1) {
    return dim_t{1, no_dim, no_dim, no_dim};
  } else if (ndim == 2) {
    return dim_t{shape[1], 1, no_dim, no_dim};
  } else if (ndim == 3) {
    return dim_t{shape[1] * shape[2], shape[2], 1, no_dim};
  } else if (ndim == 4) {
    return dim_t{shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3],
                 1};
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

MATHPRIM_PRIMFUNC bool is_in_bound(const dim<1> &shape, const dim<1> &sub) {
  return sub.x_ >= 0 && sub.x_ < shape.x_;
}

MATHPRIM_PRIMFUNC bool is_in_bound(const dim<2> &shape, const dim<2> &sub) {
  return sub.x_ >= 0 && sub.x_ < shape.x_ && sub.y_ >= 0 && sub.y_ < shape.y_;
}

MATHPRIM_PRIMFUNC bool is_in_bound(const dim<3> &shape, const dim<3> &sub) {
  return sub.x_ >= 0 && sub.x_ < shape.x_ && sub.y_ >= 0 && sub.y_ < shape.y_ &&
         sub.z_ >= 0 && sub.z_ < shape.z_;
}

MATHPRIM_PRIMFUNC bool is_in_bound(const dim<4> &shape, const dim<4> &sub) {
  return sub.x_ >= 0 && sub.x_ < shape.x_ && sub.y_ >= 0 && sub.y_ < shape.y_ &&
         sub.z_ >= 0 && sub.z_ < shape.z_ && sub.w_ >= 0 && sub.w_ < shape.w_;
}

MATHPRIM_PRIMFUNC dim<1> ind2sub(const dim<1> & /*shape*/, index_t index) {
  return dim<1>{index};
}

MATHPRIM_PRIMFUNC dim<2> ind2sub(const dim<2> &shape, index_t index) {
  if (shape.y_ == no_dim) {
    return dim<2>{index, no_dim};
  } else {
    return dim<2>{index / shape.y_, index % shape.y_};
  }
}

MATHPRIM_PRIMFUNC dim<3> ind2sub(const dim<3> &shape, index_t index) {
  if (shape.y_ == no_dim) {
    return dim<3>{index, no_dim, no_dim};
  } else if (shape.z_ == no_dim) {
    return dim<3>{index / shape.y_, index % shape.y_, no_dim};
  } else {
    return dim<3>{index / (shape.y_ * shape.z_), (index / shape.z_) % shape.y_,
                  index % shape.z_};
  }
}

MATHPRIM_PRIMFUNC dim<4> ind2sub(const dim<4> &shape, index_t index) {
  if (shape.y_ == no_dim) {
    return dim<4>{index, no_dim, no_dim, no_dim};
  } else if (shape.z_ == no_dim) {
    return dim<4>{index / shape.y_, index % shape.y_, no_dim, no_dim};
  } else if (shape.w_ == no_dim) {
    return dim<4>{index / (shape.y_ * shape.z_), (index / shape.z_) % shape.y_,
                  index % shape.z_, no_dim};
  } else {
    return dim<4>{index / (shape.y_ * shape.z_ * shape.w_),
                  (index / (shape.z_ * shape.w_)) % shape.y_,
                  (index / shape.w_) % shape.z_, index % shape.w_};
  }
}

///////////////////////////////////////////////////////////////////////////////
/// Iterator
/// TODO: Improve the implementation.
///////////////////////////////////////////////////////////////////////////////
template <index_t N> class dim_iterator final {
public:
  MATHPRIM_PRIMFUNC dim_iterator(const dim<N> &start, const dim<N> &end) noexcept
      : current_(start), shape_(end) {}

  MATHPRIM_PRIMFUNC dim_iterator &operator++() noexcept {
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

  MATHPRIM_PRIMFUNC dim_iterator operator++(int) noexcept {
    dim_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  MATHPRIM_PRIMFUNC const dim<N> &operator*() const noexcept { return current_; }

  MATHPRIM_PRIMFUNC bool operator!=(const dim_iterator &other) const noexcept {
    return current_ != other.current_;
  }

  MATHPRIM_PRIMFUNC bool operator==(const dim_iterator &other) const noexcept {
    return current_ == other.current_;
  }

private:
  dim<N> current_;
  dim<N> shape_;
};

/// ADL begin
template <index_t N>
MATHPRIM_PRIMFUNC dim_iterator<N> begin(const dim<N> &shape) {
  return dim_iterator<N>(dim<N>(), shape);
}

/// ADL end
template <index_t N>
MATHPRIM_PRIMFUNC dim_iterator<N> end(const dim<N> &shape) {
  return dim_iterator<N>(dim<N>(shape.size(0)), shape);
}

// merge two dim.
template <index_t N>
MATHPRIM_PRIMFUNC dim<N> merge(const dim<N> &a, const dim<N> &b,
                               const dim<N> &a_shape) {
  dim<N> output;
  for (index_t i = 0; i < N; ++i) {
    output[i] = a[i] * a_shape[i] + b[i];
  }
  return output;
}

template <index_t N>
MATHPRIM_PRIMFUNC dim<N> ceil_div(const dim<N> &a, const dim<N> &b) {
  dim<N> output;
  MATHPRIM_ASSERT(a.ndim() == b.ndim());
  for (index_t i = 0; i < a.ndim(); ++i) {
    output[i] = a[i] / b[i];
  }
  return output;
}

} // namespace mathprim
