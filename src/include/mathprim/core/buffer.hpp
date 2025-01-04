#pragma once
#include <type_traits>

#include "mathprim/core/backends/cpu.hpp"  // IWYU pragma: export
#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"

namespace mathprim {

namespace internal {

template <typename T> static constexpr bool is_trival_v = std::is_trivial_v<T>;

template <typename T>
static constexpr bool no_cvref_v
    = std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, T>;

}  // namespace internal

using buffer_deleter = void (*)(void *) noexcept;

template <typename T>
static constexpr bool is_buffer_supported_v
    = internal::is_trival_v<T> && internal::no_cvref_v<T>;

template <typename T, index_t N, device_t dev> class basic_buffer final {
public:
  static_assert(is_buffer_supported_v<T>, "Unsupported buffer type.");

  // buffer is not responsible for the allocate but responsible for the
  // deallocate.
  basic_buffer(const dim<N> &shape, const dim<N> &stride, T *data,
               device_t device, buffer_deleter deleter) :
      shape_(shape),
      stride_(stride),
      data_(data),
      device_(device),
      deleter_(deleter) {
    MATHPRIM_ASSERT(device != device_t::dynamic
                    && "Runtime device must be specified.");
    MATHPRIM_ASSERT((dev == device || dev == device_t::dynamic)
                    && "Device mismatch.");
  }

  ~basic_buffer() {
    if (data_) {
      deleter_(data_);
    }
  }

  MATHPRIM_COPY(basic_buffer, delete);

  basic_buffer(basic_buffer &&other) :
      basic_buffer(other.shape_, other.stride_, other.data_, other.device_,
                   other.deleter_) {
    other.data_ = nullptr;
  }

  basic_buffer &operator=(basic_buffer &&) = delete;  // move constructor

  // Shape of buffer.
  const dim<N> &shape() const noexcept { return shape_; }

  // Stride of buffer.
  const dim<N> &stride() const noexcept { return stride_; }

  // The valid ndim of the buffer.
  index_t ndim() const noexcept { return mathprim::ndim(shape_); }

  // The number of elements in the buffer.
  index_t numel() const noexcept { return mathprim::numel(shape_); }

  // The size of the buffer.
  index_t size() const noexcept { return numel(); }

  // The physical size of the buffer.
  index_t physical_size() const noexcept { return numel() * sizeof(T); }

  // Underlying data pointer.
  T *data() noexcept { return data_; }

  const T *data() const noexcept { return data_; }

  // Device of the buffer.
  device_t device() const noexcept { return device_; }

  // default view, implemented in buffer_view.hpp
  basic_buffer_view<T, N, dev> view();
  basic_buffer_view<const T, N, dev> view() const;
  basic_buffer_view_iterator<T, N, dev> begin();
  basic_buffer_view_iterator<const T, N, dev> begin() const;
  basic_buffer_view_iterator<T, N, dev> end();
  basic_buffer_view_iterator<const T, N, dev> end() const;

private:
  dim<N> shape_;
  dim<N> stride_;  // TODO: should the buffer have stride?
  T *data_;
  device_t device_;
  const buffer_deleter deleter_;
};

template <typename T, index_t N, device_t dev>
void memset(basic_buffer<T, N, dev> &buffer, int value) {
  if constexpr (dev == device_t::dynamic) {
    device_t dyn_dev = buffer.device();
    if (dyn_dev == device_t::cpu) {
      memset<device_t::cpu>(buffer, value);
    } else if (dyn_dev == device_t::cuda) {
      memset<device_t::cuda>(buffer, value);
    } else {
      MATHPRIM_INTERNAL_FATAL("Unsupported device.");
    }
  }
  buffer_backend_traits<T, dev>::memset(buffer.data(), value,
                                        buffer.physical_size());
}

/**
 * @brief The default creator for a buffer.
 *
 * @tparam T
 * @param shape
 * @return buffer, throw exception if failed.
 */
template <typename T, index_t N, device_t dev = device_t::cpu>
basic_buffer<T, N, dev> make_buffer(const dim<N> &shape) {
  void *ptr = buffer_backend_traits<T, dev>::alloc(shape.numel() * sizeof(T));
  dim<N> stride = make_default_stride(shape);
  return basic_buffer<T, N, dev>(shape, stride, static_cast<T *>(ptr), dev,
                                 buffer_backend_traits<T, dev>::free);
}

/**
 * @brief Alias of make_buffer.
 *
 */
template <typename T, device_t dev = device_t::cpu>
basic_buffer<T, 1, dev> make_buffer(index_t x) {
  return make_buffer<T, 1, dev>(dim<1>{x});
}

template <typename T, device_t dev = device_t::cpu, typename... Args,
          typename
          = std::enable_if_t<(std::is_convertible_v<Args, index_t> && ...)
                             && sizeof...(Args) >= 2>>
basic_buffer<T, sizeof...(Args), dev> make_buffer(Args... args) {
  return make_buffer<T, sizeof...(Args), dev>(dim<sizeof...(Args)>{args...});
}

template <typename T, index_t N, device_t dev = device_t::cpu>
basic_buffer_ptr<T, N, dev> make_buffer_ptr(const dim<N> &shape) {
  return std::make_unique<basic_buffer<T, N, dev>>(
      make_buffer<T, N, dev>(shape));
}

template <typename T, device_t dev = device_t::cpu>
basic_buffer_ptr<T, 1, dev> make_buffer_ptr(index_t x) {
  return make_buffer_ptr<T, 1, dev>(dim<1>{x});
}

template <typename T, index_t N, device_t dev = device_t::cpu, typename... Args>
basic_buffer<T, N, dev> make_buffer_ptr(Args... args) {
  static_assert(sizeof...(args) >= 2, "At least two arguments required.");
  return make_buffer_ptr<T, N, dev>(dim<N>{args...});
}

}  // namespace mathprim
