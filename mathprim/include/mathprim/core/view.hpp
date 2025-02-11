#pragma once
#include <type_traits>

#include "dim.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/core/utils/index_pack.hpp"

namespace mathprim {

///////////////////////////////////////////////////////////////////////////////
/// Operates on index_pack
///////////////////////////////////////////////////////////////////////////////
namespace internal {

// Flatten
template <typename seq>
struct flatten;
template <index_t front>
struct flatten<index_seq<front>> {
  static constexpr index_t value = front == keep_dim ? keep_dim : front;
};
template <index_t front, index_t... args>
struct flatten<index_seq<front, args...>> {
  static constexpr index_t value = front == keep_dim ? flatten<index_seq<args...>>::value : front;
};
template <typename seq>
constexpr index_t flatten_v = flatten<seq>::value;

// Slice a pack
template <index_t i, typename pack>
using slice_t = god::to_pack<god::remove_t<i, typename pack::seq>>;
template <index_t i, typename pack, index_t... seq>
constexpr MATHPRIM_PRIMFUNC slice_t<i, pack> slice_impl(const pack &full, index_seq<seq...> /*seq*/) noexcept {
  return slice_t<i, pack>((full.template get<(seq < i ? seq : seq + 1)>())...);
}

// Transpose two dimension
template <index_t i, index_t j, typename seq>
struct transpose;
template <index_t i, index_t j, index_t... svalues>
struct transpose<i, j, index_seq<svalues...>> {
  using seq = index_seq<svalues...>;
  template <index_t idx>
  static constexpr index_t v = god::get_v<seq, idx>;
  template <typename>
  struct transpose_element;
  template <index_t... idx>
  struct transpose_element<index_seq<idx...>> {
    using type = index_seq<idx == i ? v<j> : (idx == j ? v<i> : v<idx>)...>;
  };

  using type = typename transpose_element<make_index_seq<sizeof...(svalues)>>::type;
};
template <index_t i, index_t j, typename seq>
using transpose_t = typename transpose<i, j, seq>::type;
template <index_t i, index_t j, typename pack>
using transpose_impl_t = god::to_pack<internal::transpose_t<i, j, typename pack::seq>>;

template <index_t i, index_t j, typename pack, index_t... idx>
constexpr MATHPRIM_PRIMFUNC transpose_impl_t<i, j, pack> transpose_impl(const pack &src, index_seq<idx...>) {
  return transpose_impl_t<i, j, pack>{src.template get<(idx == i ? j : (idx == j ? i : idx))>()...};
}

// Extend a pack
template <typename view_type>
struct field_impl;
template <typename T, typename sshape, typename sstride, typename device>
struct field_impl<basic_view<T, sshape, sstride, device>> {
  using new_shape = god::to_pack<god::prepend_t<keep_dim, typename sshape::seq>>;
  static constexpr index_t old_stride_first = god::car_v<typename sstride::seq>;
  static constexpr index_t old_shape_first = god::car_v<typename sshape::seq>;
  static constexpr index_t new_stride_first
      = (old_shape_first == keep_dim || old_stride_first == keep_dim) ? keep_dim : old_stride_first * old_shape_first;
  using new_stride = god::to_pack<god::prepend_t<new_stride_first, typename sstride::seq>>;
  using type = basic_view<T, new_shape, new_stride, device>;
};

template <typename view_type>
using field_t = typename field_impl<view_type>::type;

}  // namespace internal

// Transpose
template <index_t i, index_t j, typename pack>
constexpr MATHPRIM_PRIMFUNC god::to_pack<internal::transpose_t<i, j, typename pack::seq>> transpose(const pack &src) {
  static_assert(i < pack::ndim && j < pack::ndim, "The indices must be less than the dimension.");
  return internal::transpose_impl<i, j>(src, make_index_seq<pack::ndim>{});
}

// Slicing
template <index_t i, index_t... svalues>
constexpr MATHPRIM_PRIMFUNC internal::slice_t<i, index_pack<svalues...>> slice(const index_pack<svalues...> &full) {
  static_assert(i < index_pack<svalues...>::ndim, "The index must be less than the dimension.");
  return internal::slice_impl<i>(full, make_index_seq<index_pack<svalues...>::ndim - 1>{});
}

///////////////////////////////////////////////////////////////////////////////
/// General template for buffer view.
///////////////////////////////////////////////////////////////////////////////
template <typename T, index_t... sshape_values, index_t... sstride_values, typename dev>
class basic_view<T, shape_t<sshape_values...>, stride_t<sstride_values...>, dev> {
public:
  using sshape = shape_t<sshape_values...>;
  using sstride = stride_t<sstride_values...>;
  static constexpr index_t ndim = sshape::ndim;
  static constexpr bool is_const = std::is_const_v<T>;
  using value_type = T;
  using const_type = std::add_const_t<T>;
  using byte_type = std::conditional_t<std::is_const_v<T>, const char, char>;
  using reference = T &;
  using pointer = T *;
  using indexing_type
      = std::conditional_t<ndim == 1, reference,
                           basic_view<T, internal::slice_t<0, sshape>, internal::slice_t<0, sstride>, dev>>;
  static constexpr bool is_contiguous_at_compile_time = internal::is_continuous_compile_time_v<sshape, sstride>;

  ///////////////////////////////////////////////////////////////////////////////
  /// Constructors
  ///////////////////////////////////////////////////////////////////////////////
  MATHPRIM_PRIMFUNC basic_view() noexcept : data_{nullptr} {}

  MATHPRIM_PRIMFUNC basic_view(pointer data, const sshape &shape) noexcept :
      basic_view(data, shape, make_default_stride<T>(shape)) {}

  MATHPRIM_PRIMFUNC basic_view(pointer data, const sshape &shape, const sstride &stride) noexcept :
      shape_(shape), stride_(stride), data_(data) {}

  // Allow to copy construct
  basic_view(const basic_view &) noexcept = default;
  // Allow to move construct
  basic_view(basic_view &&) noexcept = default;

  template <typename Scalar2, typename sshape2, typename sstride2,
            typename = std::enable_if_t<std::is_same_v<std::decay_t<Scalar2>, std::decay_t<T>>>>
  MATHPRIM_PRIMFUNC basic_view(const basic_view<Scalar2, sshape2, sstride2, dev> &other) :  // NOLINT: implicit convert
      basic_view(other.data(), internal::safe_cast<sshape>(other.shape()),
                 internal::safe_cast<sstride>(other.stride())) {}

  // Do not allow to assign
  basic_view &operator=(const basic_view &) noexcept = default;
  basic_view &operator=(basic_view &&) noexcept = default;

  ///////////////////////////////////////////////////////////////////////////////
  /// Meta data
  ///////////////////////////////////////////////////////////////////////////////
  // Return the number of element in view
  MATHPRIM_PRIMFUNC index_t numel() const noexcept {
    return shape_.numel();
  }

  // Return shape
  MATHPRIM_PRIMFUNC const sshape &shape() const noexcept {
    return shape_;
  }

  MATHPRIM_PRIMFUNC index_t shape(index_t i) const noexcept {
    return shape_.at(i);
  }

  MATHPRIM_PRIMFUNC index_t size() const noexcept {
    return numel();
  }

  MATHPRIM_PRIMFUNC index_t size(index_t i) const noexcept {
    return shape(i);
  }

  // Return stride
  MATHPRIM_PRIMFUNC const sstride &stride() const noexcept {
    return stride_;
  }

  MATHPRIM_PRIMFUNC index_t stride(index_t i) const noexcept {
    return stride_.at(i);
  }

  // Return true if the buffer is valid
  MATHPRIM_PRIMFUNC bool valid() const noexcept {
    return data_ != nullptr;
  }

  // Return the data pointer
  MATHPRIM_PRIMFUNC pointer data() const noexcept {
    return data_;
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Meta data shortcuts
  ///////////////////////////////////////////////////////////////////////////////

  // Return true if the buffer is valid
  explicit MATHPRIM_PRIMFUNC operator bool() const noexcept {
    return valid();
  }

  // Return if the underlying data is contiguous.
  MATHPRIM_PRIMFUNC bool is_contiguous() const noexcept {
    return stride_ == make_default_stride<T>(shape_);
  }

  auto begin() const noexcept {
    return basic_view_iterator<T, sshape, sstride, dev, 0>(*this, 0);
  }

  auto end() const noexcept {
    return basic_view_iterator<T, sshape, sstride, dev, 0>(*this, shape(0));
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Data accessing.
  ///////////////////////////////////////////////////////////////////////////////
  // direct indexing.
  MATHPRIM_PRIMFUNC indexing_type operator[](index_t i) const noexcept {
    MATHPRIM_ASSERT(data_ != nullptr);
    MATHPRIM_ASSERT(i >= 0 && i < shape(0));
    if constexpr (ndim == 1) {
      if constexpr (is_contiguous_at_compile_time) {
        return data_[i];
      } else {
        index_t offset = i * stride_.template get<0>();
        return data_[offset];
      }
    } else {
      return slice<0>(i);
    }
  }

  // subscripting.
  MATHPRIM_PRIMFUNC reference operator()(const index_array<ndim> &index) const noexcept {
    MATHPRIM_ASSERT(data_ != nullptr);
    MATHPRIM_ASSERT(is_in_bound(shape_, index));
    if constexpr (ndim == 1 && is_contiguous_at_compile_time) {
      return data_[index[0]];
    } else {
      const index_t offset = sub2ind(stride_, index);
      return data_[offset];
    }
  }

  template <typename... Args,
            typename = std::enable_if_t<(std::is_integral_v<std::decay_t<Args>> && ...) && sizeof...(Args) == ndim>>
  MATHPRIM_PRIMFUNC reference operator()(Args &&...args) const noexcept {
    return operator()(index_array<ndim>(static_cast<index_t>(args)...));
  }

  /////////////////////////////////////////////////////////////////////////////
  // View Transforms
  /////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Transpose the view.
   *
   * @tparam i
   * @tparam j
   * @return The transposed view.
   */
  template <index_t i = ndim - 1, index_t j = ndim - 2>
  MATHPRIM_PRIMFUNC
      basic_view<T, internal::transpose_impl_t<i, j, sshape>, internal::transpose_impl_t<i, j, sstride>, dev>
      transpose() const noexcept {
    return {data_, ::mathprim::transpose<i, j>(shape_), ::mathprim::transpose<i, j>(stride_)};
  }

  /**
   * @brief Slice the view.
   *
   * @tparam i
   * @return The sliced view.
   */
  template <index_t i = 0>
  MATHPRIM_PRIMFUNC basic_view<T, internal::slice_t<i, sshape>, internal::slice_t<i, sstride>, dev> slice(
      index_t batch = 0) const noexcept {
    auto offset = batch * stride_.template get<i>();
    return {data_ + offset, ::mathprim::slice<i>(shape_), ::mathprim::slice<i>(stride_)};
  }

  /**
   * @brief Flatten the view.
   *
   * @return The flattened view.
   */
  MATHPRIM_PRIMFUNC basic_view<T, shape_t<internal::flatten_v<typename sshape::seq>>,
                               stride_t<god::last_v<typename sstride::seq>>, dev>
  flatten() const noexcept {
#ifndef NDEBUG
    const auto drop_last_shape = internal::slice_impl<ndim - 1>(shape_, make_index_seq<ndim - 1>{});
    const auto drop_last_stride = internal::slice_impl<ndim - 1>(stride_, make_index_seq<ndim - 1>{});
    MATHPRIM_ASSERT(make_default_stride<T>(drop_last_shape) == drop_last_stride);
#endif
    return {data_, shape_t<internal::flatten_v<typename sshape::seq>>{shape_.numel()},
            stride_t<god::last_v<typename sstride::seq>>{stride_.template get<ndim - 1>()}};
  }

  basic_view<T, shape_t<internal::flatten_v<typename sshape::seq>>, stride_t<god::last_v<typename sstride::seq>>, dev>
  safe_flatten() const {
    const auto drop_last_shape = internal::slice_impl<ndim - 1>(shape_, make_index_seq<ndim - 1>{});
    const auto drop_last_stride = internal::slice_impl<ndim - 1>(stride_, make_index_seq<ndim - 1>{});
    if (make_default_stride<T>(drop_last_shape) != drop_last_stride) {
      throw shape_error("The view is not contiguous enough for flatten.");
    }

    return flatten();
  }

  /**
   * @brief Construct a const view with same data.
   *
   * @return MATHPRIM_PRIMFUNC
   */
  MATHPRIM_PRIMFUNC basic_view<const T, sshape, sstride, dev> as_const() const {
    return {data_, shape_, stride_};
  }

private:
  sshape shape_;
  sstride stride_;
  T *data_;
};

template <typename T, typename sshape, typename sstride, typename dev, index_t batch_dim>
struct basic_view_iterator {
  using view_type = basic_view<T, sshape, sstride, dev>;
  using indexing_type = std::conditional_t<
      view_type::ndim == 1, typename view_type::reference,
      basic_view<T, internal::slice_t<batch_dim, sshape>, internal::slice_t<batch_dim, sstride>, dev>>;
  using value_type = std::remove_reference_t<indexing_type>;
  using reference = indexing_type;
  using difference_type = index_t;
  using iterator_category = std::random_access_iterator_tag;
  view_type view;
  index_t current;

  MATHPRIM_PRIMFUNC basic_view_iterator(const view_type &view, index_t current) : view(view), current(current) {}
  MATHPRIM_PRIMFUNC basic_view_iterator() noexcept : view{}, current{0} {}
  basic_view_iterator(const basic_view_iterator &) noexcept = default;
  basic_view_iterator(basic_view_iterator &&) noexcept = default;
  basic_view_iterator &operator=(const basic_view_iterator &) noexcept = default;
  basic_view_iterator &operator=(basic_view_iterator &&) noexcept = default;

  MATHPRIM_PRIMFUNC reference operator*() const noexcept {
    if constexpr (view_type::ndim == 1) {
      return view[current];
    } else {
      return view.template slice<batch_dim>(current);
    }
  }

  MATHPRIM_PRIMFUNC reference operator[](difference_type n) const noexcept {
    if constexpr (view_type::ndim == 1) {
      return view[current + n];
    } else {
      return view.template slice<batch_dim>(current + n);
    }
  }

  MATHPRIM_PRIMFUNC basic_view_iterator &operator++() noexcept {
    ++current;
    return *this;
  }

  MATHPRIM_PRIMFUNC basic_view_iterator operator++(int) noexcept {
    basic_view_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  MATHPRIM_PRIMFUNC basic_view_iterator &operator--() noexcept {
    --current;
    return *this;
  }

  MATHPRIM_PRIMFUNC basic_view_iterator operator--(int) noexcept {
    basic_view_iterator tmp = *this;
    --(*this);
    return tmp;
  }

  MATHPRIM_PRIMFUNC basic_view_iterator &operator+=(difference_type n) noexcept {
    current += n;
    return *this;
  }

  MATHPRIM_PRIMFUNC basic_view_iterator operator+(difference_type n) const noexcept {
    basic_view_iterator tmp = *this;
    tmp += n;
    return tmp;
  }

  MATHPRIM_PRIMFUNC basic_view_iterator &operator-=(difference_type n) noexcept {
    current -= n;
    return *this;
  }

  MATHPRIM_PRIMFUNC basic_view_iterator operator-(difference_type n) const noexcept {
    basic_view_iterator tmp = *this;
    tmp -= n;
    return tmp;
  }

  MATHPRIM_PRIMFUNC bool operator==(const basic_view_iterator &other) const noexcept {
    return current == other.current && view.data() == other.view.data();
  }

  MATHPRIM_PRIMFUNC bool operator!=(const basic_view_iterator &other) const noexcept {
    return !(*this == other);
  }

  MATHPRIM_PRIMFUNC bool operator<(const basic_view_iterator &other) const noexcept {
    return current < other.current;
  }

  MATHPRIM_PRIMFUNC bool operator<=(const basic_view_iterator &other) const noexcept {
    return current <= other.current;
  }

  MATHPRIM_PRIMFUNC bool operator>(const basic_view_iterator &other) const noexcept {
    return current > other.current;
  }

  MATHPRIM_PRIMFUNC bool operator>=(const basic_view_iterator &other) const noexcept {
    return current >= other.current;
  }

  MATHPRIM_PRIMFUNC difference_type operator-(const basic_view_iterator &other) const noexcept {
    return current - other.current;
  }
};

// n + I
template <typename T, typename sshape, typename sstride, typename dev, index_t batch_dim>
MATHPRIM_PRIMFUNC basic_view_iterator<T, sshape, sstride, dev, batch_dim> operator+(
    index_t n, const basic_view_iterator<T, sshape, sstride, dev, batch_dim> &it) {
  return it + n;
}

template <typename T, typename sshape, typename device>
using continuous_view = basic_view<T, sshape, default_stride_t<sshape>, device>;
template <typename base_view>
using field_t = internal::field_t<base_view>;

///////////////////////////////////////////////////////////////////////////////
/// Create views
///////////////////////////////////////////////////////////////////////////////

template <typename device = device::cpu, typename T, typename sshape, typename sstride = default_stride_t<sshape>>
basic_view<T, sshape, sstride, device> view(T *data, const sshape &shape) {
  return basic_view<T, sshape, sstride, device>(data, shape);
}

template <typename device = device::cpu, typename T, typename sshape, typename sstride = default_stride_t<sshape>>
basic_view<T, sshape, sstride, device> view(T *data, const sshape &shape, const sstride &stride) {
  return basic_view<T, sshape, sstride, device>(data, shape, stride);
}

///////////////////////////////////////////////////////////////////////////////
/// Memcpy between continuous views.
///////////////////////////////////////////////////////////////////////////////
template <typename T1, typename sshape1, typename sstride1, typename dev1, typename T2, typename sshape2,
          typename sstride2, typename dev2>
void copy(basic_view<T1, sshape1, sstride1, dev1> dst, basic_view<T2, sshape2, sstride2, dev2> src) {
  if (!src.is_contiguous() || !dst.is_contiguous()) {
    throw std::runtime_error("The source or destination view is not contiguous.");
  }

  const auto total = src.numel() * sizeof(T1);
  const auto avail = dst.numel() * sizeof(T2);
  if (avail < total) {
    throw std::runtime_error("The destination buffer is too small.");
  }
  device::basic_memcpy<dev2, dev1>{}(dst.data(), src.data(), total);
}

}  // namespace mathprim
