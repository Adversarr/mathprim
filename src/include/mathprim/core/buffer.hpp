#pragma once
#include <type_traits>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"

namespace mathprim {

namespace internal {

template <typename T>
static constexpr bool is_trival_v = std::is_trivial_v<T>;

template <typename T>
static constexpr bool no_cvref_v = std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, T>;

}  // namespace internal

using buffer_deleter = void (*)(void *) noexcept;

template <typename T>
static constexpr bool is_buffer_supported_v = internal::is_trival_v<T> && internal::no_cvref_v<T>;

template <typename T>
class basic_buffer final {
public:
  static_assert(is_buffer_supported_v<T>, "Unsupported buffer type.");

  // buffer is not responsible for the allocate but responsible for the deallocate.
  basic_buffer(const dim_t &shape, const dim_t &stride, T *data, device_t device,
               buffer_deleter deleter)
      : shape_(shape), stride_(stride), data_(data), device_(device), deleter_(deleter) {}

  ~basic_buffer() { deleter_(data_); }

  MATHPRIM_COPY(basic_buffer, delete);
  MATHPRIM_MOVE(basic_buffer, default);  // move constructor

  // Shape of buffer.
  const dim_t &shape() const noexcept { return shape_; }

  // Stride of buffer.
  const dim_t &stride() const noexcept { return stride_; }

  // The valid ndim of the buffer.
  index_t ndim() const noexcept { return mathprim::ndim(shape_); }

  // The number of elements in the buffer.
  size_t numel() const noexcept { return mathprim::numel(shape_); }

  // The size of the buffer.
  size_t size() const noexcept { return numel(); }

  // The physical size of the buffer.
  size_t physical_size() const noexcept { return numel() * sizeof(T); }

  // Underlying data pointer.
  T *data() noexcept { return data_; }

  const T *data() const noexcept { return data_; }

  // Device of the buffer.
  device_t device() const noexcept { return device_; }

  // default view, implemented in buffer_view.hpp
  template <index_t N = max_supported_dim, device_t dev = device_t::dynamic>
  basic_buffer_view<T, N, dev> view();
  template <index_t N = max_supported_dim, device_t dev = device_t::dynamic>
  basic_buffer_view<const T, N, dev> view() const;

private:
  dim_t shape_;
  dim_t stride_;
  T *data_;
  device_t device_;
  const buffer_deleter deleter_;
};

template <device_t dev, typename T>
void memset(basic_buffer<T>& buffer, int value) {
  if constexpr (dev == device_t::dynamic) {
    device_t dyn_dev = buffer.device();
    if (dyn_dev == device_t::cpu) {
      memset<device_t::cpu>(buffer, value);
    } else if (dyn_dev == device_t::cuda) {
      memset<device_t::cuda>(buffer, value);
    } else {
      MATHPRIM_INTERNAL_FATAL();
      MATHPRIM_UNREACHABLE();
    }
  }
  backend_traits<T, dev>::memset(buffer.data(), value, buffer.physical_size());
}

/**
 * @brief The default creator for a buffer.
 *
 * @tparam T
 * @param shape
 * @return buffer, throw exception if failed.
 */
template <typename T, device_t dev = device_t::cpu>
basic_buffer<T> make_buffer(const dim_t &shape) {
  void *ptr = backend_traits<T, dev>::alloc(shape.numel() * sizeof(T));
  return basic_buffer<T>(shape, make_default_stride(shape), static_cast<T *>(ptr), dev,
                         backend_traits<T, dev>::free);
}

/**
 * @brief Alias of make_buffer.
 *
 */
template <typename T, device_t dev = device_t::cpu>
basic_buffer<T> make_buffer(index_t x) {
  return make_buffer<T, dev>(dim_t{x});
}

template <typename T, device_t dev = device_t::cpu>
basic_buffer<T> make_buffer_ptr(const dim_t &shape) {
  return std::make_unique<basic_buffer<T>>(make_buffer<T, dev>(shape));
}

template <typename T, device_t dev = device_t::cpu>
basic_buffer<T> make_buffer_ptr(index_t x) {
  return make_buffer_ptr<T, dev>(dim_t{x});
}

}  // namespace mathprim
