#pragma once
#include <memory>
#include <type_traits>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"

namespace mathprim {

using buffer_deleter = void (*)(void *) noexcept;

template <typename T>
class basic_buffer {
public:
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

  size_t size() const noexcept { return numel(); }

  // Underlying data pointer.
  T *data() noexcept { return data_; }

  const T *data() const noexcept { return data_; }

  // Device of the buffer.
  device_t device() const noexcept { return device_; }

private:
  dim_t shape_;
  dim_t stride_;
  T *data_;
  device_t device_;
  const buffer_deleter deleter_;
};

using f32_buffer = basic_buffer<float>;
using float_buffer = f32_buffer;
using f64_buffer = basic_buffer<double>;
using double_buffer = f64_buffer;
using index_buffer = basic_buffer<index_t>;

// NOTE: check size.
// static_assert(sizeof(f32_buffer) == sizeof(float_buffer),
//               "f32_buffer and float_buffer should have the same size.");

template <typename T>
using basic_buffer_ptr = std::unique_ptr<basic_buffer<T>>;

template <typename T, device_t dev>
struct buffer_traits {
  static_assert(!std::is_same_v<T, T>, "Unsupported device type.");

  static constexpr size_t alloc_alignment = 0;  ///< The alignment of the buffer.

  static constexpr basic_buffer<T> make_buffer(const dim_t & /* shape */) {
    MATHPRIM_UNREACHABLE();
  }
};

}  // namespace mathprim
