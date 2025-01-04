#pragma once

#include <stdexcept>
#include <type_traits>

#include "dim.hpp"
#include "mathprim/core/buffer.hpp"  // IWYU pragma: keep
#include "mathprim/core/defines.hpp"

namespace mathprim {

class reshape_error : public std::runtime_error {
public:
  explicit reshape_error(const std::string &msg) : std::runtime_error(msg) {}
};

namespace internal {

template <index_t M, index_t N>
inline dim<N> determine_reshape_shape(const dim<M> &from, const dim<N> &to) {
  dim<N> new_shape = to;
  index_t total_to = 1;
  index_t keep_dim_dim = -1;
  for (index_t i = 0; i < N; i++) {
    if (to[i] == -1) {
#ifndef __CUDA_ARCH__
      if (!(keep_dim_dim >= 0)) {
        throw reshape_error("Only one dimension can be -1.");
      }
#else
      MATHPRIM_ASSERT(keep_dim_dim < 0 && "Only one dimension can be -1.");
#endif

      keep_dim_dim = i;
    } else {
      total_to *= to[i];
    }
  }

  index_t total_from = from.numel();  // from is a valid shape.
  if (keep_dim_dim >= 0) {
    new_shape[keep_dim_dim] = total_from / total_to;
    total_to *= new_shape[keep_dim_dim];
  }

#ifndef __CUDA_ARCH__
  if (total_to != total_from) {
    throw reshape_error("Total number of elements must be the same.");
  }
#else
  MATHPRIM_ASSERT(total_to == total_from
                  && "Total number of elements must be the same.");
#endif
  return new_shape;
}

template <typename T> MATHPRIM_PRIMFUNC void swap_(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

}  // namespace internal

// General template for buffer view.
template <typename T, index_t N, device_t dev> class basic_buffer_view final {
public:
  using value_type = T;
  using const_type = std::add_const_t<T>;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = basic_buffer_view_iterator<T, N, dev>;
  using const_iterator = basic_buffer_view_iterator<const T, N, dev>;

  ///////////////////////////////////////////////////////////////////////////////
  /// Constructors
  ///////////////////////////////////////////////////////////////////////////////

  // default
  MATHPRIM_PRIMFUNC basic_buffer_view() noexcept : data_{nullptr} {}

  MATHPRIM_PRIMFUNC basic_buffer_view(const dim<N> &shape,
                                      pointer data) noexcept :
      basic_buffer_view(shape, make_default_stride(shape), data, dev) {}

  MATHPRIM_PRIMFUNC basic_buffer_view(const dim<N> &shape, pointer data,
                                      device_t dyn_dev) noexcept :
      basic_buffer_view(shape, make_default_stride(shape), data, dyn_dev) {}

  MATHPRIM_PRIMFUNC basic_buffer_view(const dim<N> &shape, const dim<N> &stride,
                                      pointer data, device_t dyn_dev) noexcept :
      shape_(shape), stride_(stride), data_(data), dyn_dev_(dyn_dev) {
    MATHPRIM_ASSERT(dyn_dev != device_t::dynamic
                    && "Runtime device must be specified.");
    MATHPRIM_ASSERT((dev == device_t::dynamic || dyn_dev == dev)
                    && "Device mismatch.");
  }

  template <typename T2, index_t N2, device_t dev2>
  MATHPRIM_PRIMFUNC basic_buffer_view(  // NOLINT
      basic_buffer_view<T2, N2, dev2> other) noexcept :
      basic_buffer_view(dim<N>(other.shape()), dim<N>(other.stride()),
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
  MATHPRIM_PRIMFUNC index_t numel() const noexcept {
    return mathprim::numel<N>(shape_);
  }

  // Returns the actual dimension of the buffer.
  MATHPRIM_PRIMFUNC index_t ndim() const noexcept {
    return mathprim::ndim<N>(shape_);
  }

  // Return shape
  MATHPRIM_PRIMFUNC const dim<N> &shape() const noexcept { return shape_; }

  MATHPRIM_PRIMFUNC index_t shape(index_t i) const {
    if (i < 0) {
      return shape_[ndim() + i];
    } else {
      return shape_[i];
    }
  }

  MATHPRIM_PRIMFUNC index_t size() const { return numel(); }

  MATHPRIM_PRIMFUNC index_t size(index_t i) const { return shape(i); }

  // Return stride
  MATHPRIM_PRIMFUNC const dim<N> &stride() const noexcept { return stride_; }

  MATHPRIM_PRIMFUNC index_t stride(index_t i) const {
    if (i < 0) {
      return stride_[ndim() + i];
    } else {
      return stride_[i];
    }
  }

  // Return true if the buffer is valid
  MATHPRIM_PRIMFUNC bool valid() const noexcept { return data_ != nullptr; }

  // Return the data pointer
  MATHPRIM_PRIMFUNC pointer data() const noexcept { return data_; }

  ///////////////////////////////////////////////////////////////////////////////
  /// Meta data shortcuts
  ///////////////////////////////////////////////////////////////////////////////

  // Return true if the buffer is valid
  explicit MATHPRIM_PRIMFUNC operator bool() const noexcept { return valid(); }

  // Return if the underlying data is contiguous.
  MATHPRIM_PRIMFUNC bool is_contiguous() const noexcept {
    return stride() == make_default_stride(shape_);
  }

  // TODO: Maybe we should iterate over internal data?
  auto begin() const noexcept { return iterator(*this, 0); }

  auto end() const noexcept { return iterator(*this, size(0)); }

  ///////////////////////////////////////////////////////////////////////////////
  /// Data accessing.
  ///////////////////////////////////////////////////////////////////////////////

  // if your buffer_view contains only one element, this function can help you
  // access the element directly.
  MATHPRIM_PRIMFUNC reference operator*() noexcept {
    MATHPRIM_ASSERT(data_ != nullptr && "Buffer is not valid.");
    return *data_;
  }

  // direct access to data, ignores stride
  MATHPRIM_PRIMFUNC reference operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(data_ != nullptr && "Buffer is not valid.");
    MATHPRIM_ASSERT(is_contiguous() && "Buffer is not contiguous.");
    MATHPRIM_ASSERT(i >= 0 && i < numel());
    return data_[i];
  }

  // subscripting.
  MATHPRIM_PRIMFUNC reference operator()(const dim<N> &index) const noexcept {
    MATHPRIM_ASSERT(data_ != nullptr);
    check_in_bounds(shape_, index);
    size_t offset = sub2ind(stride_, index);
    return data_[offset];
  }

  template <typename... Args,
            typename
            = std::enable_if_t<(std::is_convertible_v<Args, index_t> && ...)>>
  MATHPRIM_PRIMFUNC reference operator()(Args &&...args) const noexcept {
    return operator()(dim<N>(static_cast<index_t>(args)...));
  }

  // Reshape the buffer view.
  template <index_t M>
  MATHPRIM_PRIMFUNC basic_buffer_view<T, M, dev> view(const dim<M> &new_shape) {
    dim<M> target_shape = internal::determine_reshape_shape(shape_, new_shape);
    return basic_buffer_view<T, M, dev>(target_shape, data_, dyn_dev_);
  }

  template <index_t M>
  MATHPRIM_PRIMFUNC basic_buffer_view<const T, M, dev> view(
      const dim<M> &new_shape) const {
    dim<M> target_shape = internal::determine_reshape_shape(shape_, new_shape);
    return basic_buffer_view<T, M, dev>(target_shape, data_, dyn_dev_);
  }

  template <index_t M> MATHPRIM_PRIMFUNC basic_buffer_view<T, M, dev> view() {
    dim<M> target_shape{1};  // Initialize to 1, nodim...
    for (index_t i = 1; i < M; i++) {
      target_shape[i] = shape_[N - M + i];
    }
    target_shape[0] = numel() / target_shape.numel();
    return basic_buffer_view<T, M, dev>(target_shape, data_, dyn_dev_);
  }

  template <index_t M>
  MATHPRIM_PRIMFUNC basic_buffer_view<const T, M, dev> view() const {
    dim<M> target_shape{1};  // Initialize to 1, nodim...
    for (index_t i = 1; i < M; i++) {
      target_shape[i] = shape_[N - M + i];
    }
    target_shape[0] = numel() / target_shape.numel();
    return basic_buffer_view<T, M, dev>(target_shape, data_, dyn_dev_);
  }

  // TODO:
  // 1. subview

  template <index_t batch_dim = 0>
  MATHPRIM_PRIMFUNC basic_buffer_view<T, N - 1, dev> slice(
      index_t i) const noexcept {
    MATHPRIM_ASSERT(i >= 0 && i < shape_[batch_dim]);
    dim<N - 1> sshape;
    dim<N - 1> sstride;

    for (index_t j = 0; j < batch_dim; j++) {
      sshape[j] = shape_[j];
      sstride[j] = stride_[j];
    }

    for (index_t j = batch_dim; j < N - 1; j++) {
      sshape[j] = shape_[j + 1];
      sstride[j] = stride_[j + 1];
    }

    return {sshape, sstride, (data_ + stride_[batch_dim] * i), dyn_dev_};
  }

  MATHPRIM_PRIMFUNC basic_buffer_view<T, 1, dev> flatten() {
    return view<1>(dim<1>{numel()});
  }

  MATHPRIM_PRIMFUNC basic_buffer_view<const T, 1, dev> flatten() const {
    return view<1>(dim<1>{numel()});
  }

  MATHPRIM_PRIMFUNC basic_buffer_view<T, N, dev> transpose(index_t i,
                                                           index_t j) {
    dim<N> new_shape = shape_, new_stride = stride_;
    if (i < 0) {
      i += N;
    }
    if (j < 0) {
      j += N;
    }
    internal::swap_(new_shape[i], new_shape[j]);
    internal::swap_(new_stride[i], new_stride[j]);
    return basic_buffer_view<T, N, dev>(new_shape, new_stride, data_, dyn_dev_);
  }

  template <device_t new_dev>
  MATHPRIM_PRIMFUNC basic_buffer_view<T, N, new_dev> as() {
    MATHPRIM_ASSERT(dyn_dev_ == new_dev);
    return basic_buffer_view<T, N, new_dev>(shape_, stride_, data_, new_dev);
  }

  template <device_t new_dev>
  MATHPRIM_PRIMFUNC basic_buffer_view<const T, N, new_dev> as() const {
    MATHPRIM_ASSERT(dyn_dev_ == new_dev);
    return basic_buffer_view<const T, N, new_dev>(shape_, stride_, data_,
                                                  new_dev);
  }

  MATHPRIM_PRIMFUNC basic_buffer_view<const T, N, dev> as_const() const {
    return basic_buffer_view<const T, N, dev>(shape_, stride_, data_, dyn_dev_);
  }

private:
  dim<N> shape_;
  dim<N> stride_;
  T *data_;
  device_t dyn_dev_;  // TODO: EBO if necessary
};

template <typename T, index_t N, device_t dev>
basic_buffer_view<T, N, dev> basic_buffer<T, N, dev>::view() {
  return basic_buffer_view<T, N, dev>(shape_, stride_, data_, device_);
}

template <typename T, index_t N, device_t dev>
basic_buffer_view<const T, N, dev> basic_buffer<T, N, dev>::view() const {
  return basic_buffer_view<const T, N, dev>(shape_, stride_, data_, device_);
}

template <typename T, index_t N, device_t dev>
class basic_buffer_view_iterator {
public:
  // NOTE: currently we do not allow you to use the iterator on the device.
  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator(
      basic_buffer_view<T, N, dev> view, index_t index) :
      view_(view), current_(index) {}

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator(
      const basic_buffer_view_iterator<T, N, dev> &other)
      = default;

  MATHPRIM_FORCE_INLINE
  basic_buffer_view<T, N - 1, dev> operator*() const {
    return view_.template slice<0>(current_);
  }

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator &operator++() {
    ++current_;
    return *this;
  }

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator operator++(int) {
    basic_buffer_view_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator &operator--() {
    --current_;
    return *this;
  }

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator operator--(int) {
    basic_buffer_view_iterator tmp = *this;
    --(*this);
    return tmp;
  }

  MATHPRIM_FORCE_INLINE bool operator==(
      const basic_buffer_view_iterator &other) const {
    return current_ == other.current_;
  }

  MATHPRIM_FORCE_INLINE bool operator!=(
      const basic_buffer_view_iterator &other) const {
    return !(*this == other);
  }

private:
  basic_buffer_view<T, N, dev> view_;
  index_t current_;
};

template <typename T, device_t dev>
class basic_buffer_view_iterator<T, 1, dev> {
public:
  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator(
      basic_buffer_view<T, 1, dev> view, index_t index) :
      view_(view), current_(index) {}

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator(
      const basic_buffer_view_iterator<T, 1, dev> &other)
      = default;

  MATHPRIM_FORCE_INLINE T &operator*() const { return view_.data()[current_]; }

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator &operator++() {
    ++current_;
    return *this;
  }

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator operator++(int) {
    basic_buffer_view_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator &operator--() {
    --current_;
    return *this;
  }

  MATHPRIM_FORCE_INLINE basic_buffer_view_iterator operator--(int) {
    basic_buffer_view_iterator tmp = *this;
    --(*this);
    return tmp;
  }

  MATHPRIM_FORCE_INLINE bool operator==(
      const basic_buffer_view_iterator &other) const {
    return current_ == other.current_;
  }

  MATHPRIM_FORCE_INLINE bool operator!=(
      const basic_buffer_view_iterator &other) const {
    return !(*this == other);
  }

private:
  basic_buffer_view<T, 1, dev> view_;
  index_t current_;
};

}  // namespace mathprim
