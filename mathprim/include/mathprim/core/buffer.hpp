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

template <typename from, typename to>
struct can_cast;
template <index_t... from_values, index_t... to_values>
struct can_cast<index_pack<from_values...>, index_pack<to_values...>> {
  static constexpr bool value = ((from_values == to_values || from_values == keep_dim || to_values == keep_dim) && ...);
};

template <typename from, typename to>
static constexpr bool can_cast_v = can_cast<from, to>::value;

}  // namespace internal

template <typename T, index_t... sshape_values, index_t... sstride_values, typename dev>
class basic_buffer<T, index_pack<sshape_values...>, index_pack<sstride_values...>, dev> {
public:
  using sshape = index_pack<sshape_values...>;
  using sstride = index_pack<sstride_values...>;
  static_assert(internal::is_buffer_supported_v<T>, "Unsupported buffer type.");
  template <typename, typename, typename, typename>
  friend class basic_buffer;  // ok, they are friends.

  // not responsible for the allocation but responsible for deallocation
  basic_buffer(T *data, const sshape &shape) : basic_buffer(data, shape, make_default_stride<T>(shape)) {}
  basic_buffer(T *data, const sshape &shape, const sstride &stride) : shape_(shape), stride_(stride), data_(data) {}
  // basic_buffer(basic_buffer &&other) noexcept : shape_(other.shape_), stride_(other.stride_), data_(other.data_) {
  //   other.data_ = nullptr;
  // }

  template <typename sshape2, typename sstride2,
            typename
            = std::enable_if_t<internal::can_cast_v<sshape2, sshape> && internal::can_cast_v<sstride2, sstride>>>
  basic_buffer(basic_buffer<T, sshape2, sstride, dev> &&other) :  // NOLINT: explicit
      shape_(other.shape()), stride_(other.stride()), data_(other.data()) {
    other.data_ = nullptr;
  }

  ~basic_buffer() {
    if (data_) {
      dev{}.free(data_);
      data_ = nullptr;
    }
  }
  MATHPRIM_INTERNAL_COPY(basic_buffer, delete);
  basic_buffer &operator=(basic_buffer &&) = delete;  // move constructor

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

/**
 * @brief The default creator for a buffer.
 *
 * @tparam T
 * @param shape
 * @return buffer, throw exception if failed.
 */
template <typename T, typename dev = device::cpu, typename sshape>
basic_buffer<T, sshape, internal::default_stride_t<T, sshape>, dev> make_buffer(const sshape &shape) {
  auto ptr = static_cast<T *>(dev{}.malloc(sizeof(T) * mathprim::numel(shape)));
  return basic_buffer<T, sshape, internal::default_stride_t<T, sshape>, dev>(ptr, shape);
}

}  // namespace mathprim
