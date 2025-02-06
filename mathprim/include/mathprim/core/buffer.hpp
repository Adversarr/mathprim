#pragma once
#include <type_traits>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"
#include "mathprim/core/view.hpp"

namespace mathprim {

namespace internal {

template <typename T>
static constexpr bool is_trival_v = std::is_trivial_v<T>;

template <typename T>
static constexpr bool no_cvref_v = std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, T>;

template <typename T>
static constexpr bool is_buffer_supported_v = internal::is_trival_v<T> && internal::no_cvref_v<T>;

template <typename sshape_from, typename sstride_from, typename sshape_to, typename sstride_to>
static constexpr bool is_buffer_castable_v
    = is_castable_v<sshape_from, sshape_to> && is_castable_v<sstride_from, sstride_to>;
}  // namespace internal

template <typename T, index_t... sshape_values, index_t... sstride_values, typename dev>
class basic_buffer<T, index_pack<sshape_values...>, index_pack<sstride_values...>, dev> {
public:
  using sshape = index_pack<sshape_values...>;
  using sstride = index_pack<sstride_values...>;
  static_assert(internal::is_buffer_supported_v<T>, "Unsupported buffer type.");

  template <typename, typename, typename, typename>
  friend class basic_buffer;  // ok, they are friends.

  // Move constructor: allow to cast from a buffer with same shape and stride at runtime.
  template <typename sshape2, typename sstride2,
            typename = std::enable_if_t<internal::is_buffer_castable_v<sshape2, sstride2, sshape, sstride>>>
  basic_buffer(basic_buffer<T, sshape2, sstride2, dev> &&other) :  // NOLINT: explicit
      shape_(other.shape()), stride_(other.stride()), data_(other.data()) {
    other.data_ = nullptr;
  }

  // not responsible for the allocation but responsible for deallocation
  basic_buffer(T *data, const sshape &shape) : basic_buffer(data, shape, make_default_stride<T>(shape)) {}
  basic_buffer(T *data, const sshape &shape, const sstride &stride) : shape_(shape), stride_(stride), data_(data) {}

  // Disable copy constructor and all assignment.
  basic_buffer(const basic_buffer &) = delete;
  basic_buffer &operator=(const basic_buffer &) = delete;
  basic_buffer &operator=(basic_buffer &&) = delete;

  // Deleter.
  ~basic_buffer() {
    if (data_) {
      dev{}.free(data_);
      data_ = nullptr;
    }
  }

  // swap

  template <typename sshape2, typename sstride2,
            typename = std::enable_if_t<internal::is_buffer_castable_v<sshape2, sstride2, sshape, sstride>>>
  void swap(basic_buffer<T, sshape2, sstride2, dev> &other) noexcept {
    std::swap(data_, other.data_);
    internal::swap_impl(shape_.dyn_, other.shape_.dyn_);
    internal::swap_impl(stride_.dyn_, other.stride_.dyn_);
  }

  // Shape of buffer.
  const sshape &shape() const noexcept {
    return shape_;
  }

  index_t shape(index_t i) const noexcept {
    return shape_.at(i);
  }

  // Stride of buffer.
  const sstride &stride() const noexcept {
    return stride_;
  }

  index_t stride(index_t i) const noexcept {
    return stride_.at(i);
  }

  // The valid ndim of the buffer.
  index_t ndim() const noexcept {
    return mathprim::ndim(shape_);
  }

  // The number of elements in the buffer.
  index_t numel() const noexcept {
    return mathprim::numel(shape_);
  }

  // The size of the buffer.
  index_t size() const noexcept {
    return numel();
  }

  // The physical size of the buffer.
  index_t physical_size() const noexcept {
    return stride_.template get<0>() * shape_.template get<0>();
  }

  // Underlying data pointer.
  T *data() noexcept {
    return data_;
  }

  const T *data() const noexcept {
    return data_;
  }

  using view_type = basic_view<T, sshape, sstride, dev>;
  using const_view_type = basic_view<const T, sshape, sstride, dev>;
  using iterator = dimension_iterator<T, sshape, sstride, dev>;
  using const_iterator = dimension_iterator<const T, sshape, sstride, dev>;

  // default view, implemented in view.hpp
  view_type view() noexcept {
    return view_type(data_, shape_, stride_);
  }
  const_view_type view() const noexcept {
    return const_view();
  }

  const_view_type const_view() const noexcept {
    return const_view_type(data_, shape_, stride_);
  }

  iterator begin() noexcept {
    return view().begin();
  }

  const_iterator begin() const noexcept {
    return view().begin();
  }

  iterator end() noexcept {
    return view().end();
  }

  const_iterator end() const noexcept {
    return view().end();
  }

private:
  sshape shape_;
  sstride stride_;
  T *data_;
};

template <typename T, typename sshape, typename dev>
using continuous_buffer = basic_buffer<T, sshape, internal::default_stride_t<T, sshape>, dev>;

/**
 * @brief The default creator for a buffer.
 *
 * @tparam T
 * @param shape
 * @return buffer, throw exception if failed.
 */
template <typename T, typename dev = device::cpu, typename sshape>
continuous_buffer<T, sshape, dev> make_buffer(const sshape &shape) {
  auto ptr = static_cast<T *>(dev{}.malloc(sizeof(T) * mathprim::numel(shape)));
  return basic_buffer<T, sshape, internal::default_stride_t<T, sshape>, dev>(ptr, shape);
}

/**
 * @brief Create a continuous buffer, but no static information.
 *
 * @tparam T [TODO:tparam]
 * @tparam Integers [TODO:tparam]
 * @param shape [TODO:parameter]
 * @return [TODO:return]
 */
template <typename T, typename dev = device::cpu, typename... Args,
          typename = std::enable_if_t<(internal::can_hold_v<Args> && ...)>>
auto make_buffer(Args... shape) {
  return make_buffer<T, dev>(make_shape(shape...));
}

}  // namespace mathprim
