#pragma once

#include <type_traits>

#include "dim.hpp"
#include "mathprim/core/buffer.hpp"  // IWYU pragma: keep
#include "mathprim/core/defines.hpp"

namespace mathprim {

// General template for buffer view.
template <typename T, index_t N, device_t dev>
class basic_buffer_view final {
public:
  using value_type = T;
  using const_type = std::add_const_t<T>;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;

  ///////////////////////////////////////////////////////////////////////////////
  /// Constructors
  ///////////////////////////////////////////////////////////////////////////////

  // default
  MATHPRIM_PRIMFUNC basic_buffer_view() noexcept : data_{nullptr} {}

  MATHPRIM_PRIMFUNC basic_buffer_view(const dim<N> &shape, pointer data,
                                      device_t dyn_dev) noexcept
      : basic_buffer_view(shape, make_default_stride(shape), data, dyn_dev) {}

  MATHPRIM_PRIMFUNC basic_buffer_view(const dim<N> &shape, const dim<N> &stride,
                                      pointer data, device_t dyn_dev) noexcept
      : shape_(shape), stride_(stride), data_(data), dyn_dev_(dyn_dev) {
    MATHPRIM_ASSERT(dyn_dev != device_t::dynamic
                    && "Runtime device must be specified.");
    MATHPRIM_ASSERT((dev == device_t::dynamic || dyn_dev == dev)
                    && "Device mismatch.");
  }

  template <typename T2, index_t N2, device_t dev2>
  MATHPRIM_PRIMFUNC basic_buffer_view(
      basic_buffer_view<T2, N2, dev2> other) noexcept  // NOLINT
      : basic_buffer_view(dim<N>(other.shape()), dim<N>(other.stride()),
                          other.data(), other.device()) {
    // Although the constructor of dim<N> will test it, we this assert here to
    // make sure.
    MATHPRIM_ASSERT(other.ndim() <= N && "Assigning to a smaller buffer view.");
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Meta data
  ///////////////////////////////////////////////////////////////////////////////

  // Return the device of buffer
  MATHPRIM_PRIMFUNC device_t device() const noexcept { return dyn_dev_; }

  // Return the number of element in view
  MATHPRIM_PRIMFUNC size_t numel() const noexcept {
    return mathprim::numel<N>(shape_);
  }

  // Returns the actual dimension of the buffer.
  MATHPRIM_PRIMFUNC index_t ndim() const noexcept {
    return mathprim::ndim<N>(shape_);
  }

  // Return shape
  MATHPRIM_PRIMFUNC const dim<N> &shape() const noexcept { return shape_; }

  MATHPRIM_PRIMFUNC index_t shape(index_t i) {
    if (i < 0) {
      return shape_[ndim() + i];
    } else {
      return shape_[i];
    }
  }

  // Return stride
  MATHPRIM_PRIMFUNC const dim<N> &stride() const noexcept { return stride_; }

  MATHPRIM_PRIMFUNC index_t stride(index_t i) {
    if (i < 0) {
      return stride_[ndim() + i];
    } else {
      return stride_[i];
    }
  }

  // Return true if the buffer is valid
  MATHPRIM_PRIMFUNC bool valid() const noexcept { return data_ != nullptr; }

  // Return the data pointer
  MATHPRIM_PRIMFUNC pointer data() noexcept { return data_; }

  // Return the data pointer(const)
  MATHPRIM_PRIMFUNC const_pointer data() const noexcept { return data_; }

  ///////////////////////////////////////////////////////////////////////////////
  /// Meta data shortcuts
  ///////////////////////////////////////////////////////////////////////////////

  // Return true if the buffer is valid
  explicit MATHPRIM_PRIMFUNC operator bool() const noexcept { return valid(); }

  // Return if the underlying data is contiguous.
  MATHPRIM_PRIMFUNC bool is_contiguous() const noexcept {
    return stride() == make_default_stride(shape_);
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Data accessing.
  ///////////////////////////////////////////////////////////////////////////////

  // direct access to data
  MATHPRIM_PRIMFUNC reference operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(data_ != nullptr && "Buffer is not valid.");
    MATHPRIM_ASSERT(is_contiguous() && "Buffer is not contiguous.");
    MATHPRIM_ASSERT(i >= 0 && i < numel());
    return data_[i];
  }

  // subscripting.
  MATHPRIM_PRIMFUNC reference operator()(const dim<N> &index) noexcept {
    MATHPRIM_ASSERT(data_ != nullptr);
    check_in_bounds(shape_, index);
    size_t offset = sub2ind(stride_, index);
    return data_[offset];
  }

  template <typename... Args,
            typename
            = std::enable_if_t<(std::is_convertible_v<Args, index_t> && ...)>>
  MATHPRIM_PRIMFUNC reference operator()(Args &&...args) noexcept {
    return operator()(dim<N>(static_cast<index_t>(args)...));
  }

private:
  dim<N> shape_;
  dim<N> stride_;
  T *data_;
  device_t dyn_dev_;  // TODO: EBO if necessary
};

template <typename T>
template <index_t N, device_t dev>
basic_buffer_view<T, N, dev> basic_buffer<T>::view() {
  return basic_buffer_view<T, N, dev>(shape_, stride_, data_, device_);
}

template <typename T>
template <index_t N, device_t dev>
basic_buffer_view<const T, N, dev> basic_buffer<T>::view() const {
  return basic_buffer_view<const T, N, dev>(shape_, stride_, data_, device_);
}

}  // namespace mathprim
