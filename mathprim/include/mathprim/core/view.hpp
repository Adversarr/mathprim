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

// Apply byte offset
template <typename T>
constexpr MATHPRIM_PRIMFUNC T *apply_byte_offset(T *data, index_t offset) noexcept {
  if constexpr (std::is_const_v<T>) {
    return reinterpret_cast<T *>(reinterpret_cast<const char *>(data) + offset);
  } else {
    return reinterpret_cast<T *>(reinterpret_cast<char *>(data) + offset);
  }
}

// flatten
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

template <index_t i, typename pack>
using slice_t = god::to_pack<god::remove_t<i, typename pack::seq>>;

template <index_t i, typename pack, index_t... seq>
constexpr MATHPRIM_PRIMFUNC slice_t<i, pack> slice_impl(const pack &full, index_seq<seq...> /*seq*/) noexcept {
  return slice_t<i, pack>((full.template get<(seq < i ? seq : seq + 1)>())...);
}

// transpose
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
  MATHPRIM_PRIMFUNC basic_view(const basic_view<Scalar2, sshape2, sstride2, dev> &other) : // NOLINT: implicit convert
      basic_view(other.data(), internal::safe_cast<sshape>(other.shape()),
                 internal::safe_cast<sstride>(other.stride())) {}

  // Do not allow to assign
  basic_view &operator=(const basic_view &) noexcept = delete;
  basic_view &operator=(basic_view &&) noexcept = delete;

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

  // TODO: Maybe we should iterate over internal data?
  auto begin() const noexcept {
    return dimension_iterator<T, sshape, sstride, dev>(*this, 0);
  }

  auto end() const noexcept {
    return dimension_iterator<T, sshape, sstride, dev>(*this, shape(0));
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Data accessing.
  ///////////////////////////////////////////////////////////////////////////////
  // direct indexing.
  MATHPRIM_PRIMFUNC indexing_type operator[](index_t i) const noexcept {
    MATHPRIM_ASSERT(data_ != nullptr);
    MATHPRIM_ASSERT(i >= 0 && i < shape(0));
    if constexpr (ndim == 1) {
      return *internal::apply_byte_offset<T>(data_, i * stride_.template get<0>());
    } else {
      return slice<0>(i);
    }
  }

  // subscripting.
  MATHPRIM_PRIMFUNC reference operator()(const index_array<ndim> &index) const noexcept {
    MATHPRIM_ASSERT(data_ != nullptr);
    MATHPRIM_ASSERT(is_in_bound(shape_, index));
    const index_t offset = byte_offset(stride_, index);
    return *internal::apply_byte_offset<T>(data_, offset);
  }

  template <typename... Args,
            typename = std::enable_if_t<(std::is_integral_v<std::decay_t<Args>> && ...) && sizeof...(Args) == ndim>>
  MATHPRIM_PRIMFUNC reference operator()(Args &&...args) const noexcept {
    return operator()(index_array<ndim>(static_cast<index_t>(args)...));
  }

  template <index_t i = ndim - 1, index_t j = ndim - 2>
  MATHPRIM_PRIMFUNC
      basic_view<T, internal::transpose_impl_t<i, j, sshape>, internal::transpose_impl_t<i, j, sstride>, dev>
      transpose() const noexcept {
    return {data_, ::mathprim::transpose<i, j>(shape_), ::mathprim::transpose<i, j>(stride_)};
  }

  template <index_t i = 0>
  MATHPRIM_PRIMFUNC basic_view<T, internal::slice_t<i, sshape>, internal::slice_t<i, sstride>, dev> slice(
      index_t batch = 0) const noexcept {
    return {internal::apply_byte_offset(data_, batch * stride_.template get<i>()), ::mathprim::slice<i>(shape_),
            ::mathprim::slice<i>(stride_)};
  }

  MATHPRIM_PRIMFUNC basic_view<const T, sshape, sstride, dev> as_const() const {
    return {data_, shape_, stride_};
  }

private:
  const sshape shape_;
  const sstride stride_;
  T *data_;
};

template <typename T, typename sshape, typename sstride, typename dev>
struct dimension_iterator {
  using view_type = basic_view<T, sshape, sstride, dev>;
  using value_type = typename view_type::indexing_type;
  using reference = value_type;
  using pointer = value_type *;
  using difference_type = index_t;
  using iterator_category = std::random_access_iterator_tag;
  view_type view;
  index_t current;

  MATHPRIM_PRIMFUNC dimension_iterator(const view_type &view, index_t current) : view(view), current(current) {}

  MATHPRIM_PRIMFUNC reference operator*() const noexcept {
    return view[current];
  }

  MATHPRIM_PRIMFUNC reference operator[](difference_type n) const noexcept {
    return view[current + n];
  }

  MATHPRIM_PRIMFUNC dimension_iterator &operator++() noexcept {
    ++current;
    return *this;
  }

  MATHPRIM_PRIMFUNC dimension_iterator operator++(int) noexcept {
    dimension_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  MATHPRIM_PRIMFUNC dimension_iterator &operator--() noexcept {
    --current;
    return *this;
  }

  MATHPRIM_PRIMFUNC dimension_iterator operator--(int) noexcept {
    dimension_iterator tmp = *this;
    --(*this);
    return tmp;
  }

  MATHPRIM_PRIMFUNC dimension_iterator &operator+=(difference_type n) noexcept {
    current += n;
    return *this;
  }

  MATHPRIM_PRIMFUNC dimension_iterator operator+(difference_type n) const noexcept {
    dimension_iterator tmp = *this;
    tmp += n;
    return tmp;
  }

  MATHPRIM_PRIMFUNC dimension_iterator &operator-=(difference_type n) noexcept {
    current -= n;
    return *this;
  }

  MATHPRIM_PRIMFUNC dimension_iterator operator-(difference_type n) const noexcept {
    dimension_iterator tmp = *this;
    tmp -= n;
    return tmp;
  }

  MATHPRIM_PRIMFUNC bool operator==(const dimension_iterator &other) const noexcept {
    return current == other.current && view.data() == other.view.data();
  }

  MATHPRIM_PRIMFUNC bool operator!=(const dimension_iterator &other) const noexcept {
    return !(*this == other);
  }

  MATHPRIM_PRIMFUNC bool operator<(const dimension_iterator &other) const noexcept {
    return current < other.current;
  }

  MATHPRIM_PRIMFUNC bool operator<=(const dimension_iterator &other) const noexcept {
    return current <= other.current;
  }

  MATHPRIM_PRIMFUNC bool operator>(const dimension_iterator &other) const noexcept {
    return current > other.current;
  }

  MATHPRIM_PRIMFUNC bool operator>=(const dimension_iterator &other) const noexcept {
    return current >= other.current;
  }

  MATHPRIM_PRIMFUNC difference_type operator-(const dimension_iterator &other) const noexcept {
    return current - other.current;
  }
};
template <typename device = device::cpu, typename T, typename sshape,
          typename sstride = internal::default_stride_t<T, sshape>>
basic_view<T, sshape, sstride, device> make_view(T *data, const sshape &shape) {
  return basic_view<T, sshape, sstride, device>(data, shape);
}

template <typename device = device::cpu, typename T, typename sshape,
          typename sstride = internal::default_stride_t<T, sshape>>
basic_view<T, sshape, sstride, device> make_view(T *data, const sshape &shape, const sstride &stride) {
  return basic_view<T, sshape, sstride, device>(data, shape, stride);
}

}  // namespace mathprim
