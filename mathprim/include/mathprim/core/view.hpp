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
template <typename Seq>
struct flatten;
template <index_t Front>
struct flatten<index_seq<Front>> {
  static constexpr index_t value = Front == keep_dim ? keep_dim : Front;
};
template <index_t Front, index_t... Args>
struct flatten<index_seq<Front, Args...>> {
  static constexpr index_t value = Front == keep_dim ? flatten<index_seq<Args...>>::value : Front;
};
template <typename Seq>
constexpr index_t flatten_v = flatten<Seq>::value;

// Slice a pack
template <index_t I, typename Pack>
using slice_t = god::to_pack<god::remove_t<I, typename Pack::seq>>;
template <index_t I, typename Pack, index_t... Seq>
constexpr MATHPRIM_PRIMFUNC slice_t<I, Pack> slice_impl(const Pack &full, index_seq<Seq...> /*seq*/) noexcept {
  return slice_t<I, Pack>((full.template get<(Seq < I ? Seq : Seq + 1)>())...);
}

// Transpose two dimension
template <index_t I, index_t J, typename Seq>
struct transpose;
template <index_t I, index_t J, index_t... Svalues>
struct transpose<I, J, index_seq<Svalues...>> {
  using seq = index_seq<Svalues...>;
  template <index_t Idx>
  static constexpr index_t v = god::get_v<seq, Idx>;
  template <typename>
  struct transpose_element;
  template <index_t... Idx>
  struct transpose_element<index_seq<Idx...>> {
    using type = index_seq<Idx == I ? v<J> : (Idx == J ? v<I> : v<Idx>)...>;
  };

  using type = typename transpose_element<make_index_seq<sizeof...(Svalues)>>::type;
};
template <index_t I, index_t J, typename Seq>
using transpose_t = typename transpose<I, J, Seq>::type;
template <index_t I, index_t J, typename Pack>
using transpose_impl_t = god::to_pack<internal::transpose_t<I, J, typename Pack::seq>>;

template <index_t i, index_t J, typename Pack, index_t... Idx>
constexpr MATHPRIM_PRIMFUNC transpose_impl_t<i, J, Pack> transpose_impl(const Pack &src, index_seq<Idx...>) {
  return transpose_impl_t<i, J, Pack>{src.template get<(Idx == i ? J : (Idx == J ? i : Idx))>()...};
}

// Extend a pack
template <typename ViewType>
struct field_impl;
template <typename T, typename Sshape, typename Sstride, typename Device>
struct field_impl<basic_view<T, Sshape, Sstride, Device>> {
  using new_shape = god::to_pack<god::prepend_t<keep_dim, typename Sshape::seq>>;
  static constexpr index_t old_stride_first = god::car_v<typename Sstride::seq>;
  static constexpr index_t old_shape_first = god::car_v<typename Sshape::seq>;
  static constexpr index_t new_stride_first
      = (old_shape_first == keep_dim || old_stride_first == keep_dim) ? keep_dim : old_stride_first * old_shape_first;
  using new_stride = god::to_pack<god::prepend_t<new_stride_first, typename Sstride::seq>>;
  using type = basic_view<T, new_shape, new_stride, Device>;
};

template <typename ViewType>
using field_t = typename field_impl<ViewType>::type;

///////////////////////////////////////////////////////////////////////////////
/// Operations on packs.
///////////////////////////////////////////////////////////////////////////////
// Transpose
template <index_t I, index_t J, typename Pack>
constexpr MATHPRIM_PRIMFUNC god::to_pack<internal::transpose_t<I, J, typename Pack::seq>> pack_transpose(
    const Pack &src) {
  static_assert(I < Pack::ndim && J < Pack::ndim, "The indices must be less than the dimension.");
  return internal::transpose_impl<I, J>(src, make_index_seq<Pack::ndim>{});
}

// Slicing
template <index_t I, index_t... Svalues>
constexpr MATHPRIM_PRIMFUNC internal::slice_t<I, index_pack<Svalues...>> pack_slice(
    const index_pack<Svalues...> &full) {
  static_assert(I < index_pack<Svalues...>::ndim, "The index must be less than the dimension.");
  return internal::slice_impl<I>(full, make_index_seq<index_pack<Svalues...>::ndim - 1>{});
}

}  // namespace internal

///////////////////////////////////////////////////////////////////////////////
/// General template for buffer view.
///////////////////////////////////////////////////////////////////////////////
template <typename Scalar, typename Sshape, typename Sstride, typename Dev>
class basic_view {
public:
  static constexpr index_t ndim = Sshape::ndim;
  static constexpr bool is_const = std::is_const_v<Scalar>;
  using shape_at_compile_time = Sshape;
  using stride_at_compile_time = Sstride;

  using value_type = Scalar;
  using const_type = std::add_const_t<Scalar>;
  using byte_type = std::conditional_t<std::is_const_v<Scalar>, const char, char>;
  using reference = Scalar &;
  using pointer = Scalar *;
  using indexing_type
      = std::conditional_t<ndim == 1, reference,
                           basic_view<Scalar, internal::slice_t<0, Sshape>, internal::slice_t<0, Sstride>, Dev>>;

  using flatten_type = std::conditional_t<ndim == 1, basic_view<Scalar, Sshape, Sstride, Dev> /*self*/,
                                          basic_view<Scalar, shape_t<internal::flatten_v<typename Sshape::seq>>,
                                                     stride_t<god::last_v<typename Sstride::seq>>, Dev>>;

  static constexpr bool is_contiguous_at_compile_time = internal::is_continuous_compile_time_v<Sshape, Sstride>;

  ///////////////////////////////////////////////////////////////////////////////
  /// Constructors
  ///////////////////////////////////////////////////////////////////////////////
  MATHPRIM_PRIMFUNC basic_view() noexcept : data_{nullptr} {}

  MATHPRIM_PRIMFUNC basic_view(pointer data, const Sshape &shape) noexcept :
      basic_view(data, shape, make_default_stride<Scalar>(shape)) {}

  MATHPRIM_PRIMFUNC basic_view(pointer data, const Sshape &shape, const Sstride &stride) noexcept :
      shape_(shape), stride_(stride), data_(data) {}

  // Allow to copy construct
  basic_view(const basic_view &) noexcept = default;
  // Allow to move construct
  basic_view(basic_view &&) noexcept = default;

  template <typename Scalar2, typename sshape2, typename sstride2,
            typename = std::enable_if_t<std::is_same_v<std::decay_t<Scalar2>, std::decay_t<Scalar>>>>
  MATHPRIM_PRIMFUNC basic_view(const basic_view<Scalar2, sshape2, sstride2, Dev> &other) :  // NOLINT: implicit convert
      basic_view(other.data(), internal::safe_cast<Sshape>(other.shape()),
                 internal::safe_cast<Sstride>(other.stride())) {}

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
  MATHPRIM_PRIMFUNC const Sshape &shape() const noexcept {
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
  MATHPRIM_PRIMFUNC const Sstride &stride() const noexcept {
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
    return stride_ == make_default_stride<Scalar>(shape_);
  }

  auto begin() const noexcept {
    return basic_view_iterator<Scalar, Sshape, Sstride, Dev, 0>(*this, 0);
  }

  auto end() const noexcept {
    return basic_view_iterator<Scalar, Sshape, Sstride, Dev, 0>(*this, shape(0));
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Data accessing.
  ///////////////////////////////////////////////////////////////////////////////
  // direct indexing.
  MATHPRIM_PRIMFUNC indexing_type operator[](const index_t &i) const noexcept {
    MATHPRIM_ASSERT(data_ != nullptr);
    MATHPRIM_ASSERT(i >= 0 && i < shape(0));
    if constexpr (ndim == 1) {
      if constexpr (is_contiguous_at_compile_time) {
        return data_[i];
      } else {
        const index_t offset = i * stride_.template get<0>();
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
  template <index_t I = ndim - 1, index_t J = ndim - 2>
  MATHPRIM_PRIMFUNC
      basic_view<Scalar, internal::transpose_impl_t<I, J, Sshape>, internal::transpose_impl_t<I, J, Sstride>, Dev>
      transpose() const noexcept {
    return {data_, internal::pack_transpose<I, J>(shape_), internal::pack_transpose<I, J>(stride_)};
  }

  /**
   * @brief Slice the view.
   *
   * @tparam i
   * @return The sliced view.
   */
  template <index_t I = 0>
  MATHPRIM_PRIMFUNC basic_view<Scalar, internal::slice_t<I, Sshape>, internal::slice_t<I, Sstride>, Dev> slice(
      index_t batch = 0) const noexcept {
    auto offset = batch * stride_.template get<I>();
    return {data_ + offset, internal::pack_slice<I>(shape_), internal::pack_slice<I>(stride_)};
  }

  /**
   * @brief Flatten the view.
   *
   * @return The flattened view.
   */
  MATHPRIM_PRIMFUNC flatten_type flatten() const noexcept {
    if constexpr (ndim == 1) {
      return *this;
    } else {
#ifndef NDEBUG
      // Only the last dimension can have non-contiguous stride.
      const index_t stride_last = shape_.template get<ndim - 1>();
      const auto drop_last_shape = internal::slice_impl<ndim - 1>(shape_, make_index_seq<ndim - 1>{});
      const auto drop_last_stride_check = make_default_stride<Scalar>(drop_last_shape).to_array() * stride_last;
      const auto drop_last_stride = internal::slice_impl<ndim - 1>(stride_, make_index_seq<ndim - 1>{}).to_array();
      MATHPRIM_ASSERT(drop_last_stride_check == drop_last_stride && "The view is not contiguous enough for flatten.");
#endif
      return {data_, shape_t<internal::flatten_v<typename Sshape::seq>>{shape_.numel()},
              stride_t<god::last_v<typename Sstride::seq>>{stride_.template get<ndim - 1>()}};
    }
  }

  MATHPRIM_NOINLINE flatten_type safe_flatten() const {
    const auto drop_last_shape = internal::slice_impl<ndim - 1>(shape_, make_index_seq<ndim - 1>{});
    const auto drop_last_stride = internal::slice_impl<ndim - 1>(stride_, make_index_seq<ndim - 1>{});
    if (make_default_stride<Scalar>(drop_last_shape) != drop_last_stride) {
      throw shape_error("The view is not contiguous enough for flatten.");
    }

    return flatten();
  }

  /**
   * @brief Construct a const view with same data.
   *
   * @return MATHPRIM_PRIMFUNC
   */
  MATHPRIM_PRIMFUNC basic_view<const Scalar, Sshape, Sstride, Dev> as_const() const {
    return {data_, shape_, stride_};
  }

  MATHPRIM_FORCE_INLINE void swap(basic_view &other) noexcept {
    std::swap(shape_, other.shape_);
    std::swap(stride_, other.stride_);
    std::swap(data_, other.data_);
  }

private:
  Sshape shape_;
  Sstride stride_;
  Scalar *data_;
};

template <typename Scalar, typename Sshape, typename Sstride, typename Dev, index_t BatchDim>
struct basic_view_iterator {
  using view_type = basic_view<Scalar, Sshape, Sstride, Dev>;
  using indexing_type = std::conditional_t<
      view_type::ndim == 1, typename view_type::reference,
      basic_view<Scalar, internal::slice_t<BatchDim, Sshape>, internal::slice_t<BatchDim, Sstride>, Dev>>;
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
      return view.template slice<BatchDim>(current);
    }
  }

  MATHPRIM_PRIMFUNC reference operator[](difference_type n) const noexcept {
    if constexpr (view_type::ndim == 1) {
      return view[current + n];
    } else {
      return view.template slice<BatchDim>(current + n);
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
template <typename T, typename Sshape, typename Sstride, typename Dev, index_t BatchDim>
MATHPRIM_PRIMFUNC basic_view_iterator<T, Sshape, Sstride, Dev, BatchDim> operator+(
    index_t n, const basic_view_iterator<T, Sshape, Sstride, Dev, BatchDim> &it) {
  return it + n;
}

template <typename T, typename Sshape, typename Device>
using continuous_view = basic_view<T, Sshape, default_stride_t<Sshape>, Device>;
template <typename BaseView>
using field_t = internal::field_t<BaseView>;

///////////////////////////////////////////////////////////////////////////////
/// Create views
///////////////////////////////////////////////////////////////////////////////

template <typename Device = device::cpu, typename T, typename Sshape, typename Sstride = default_stride_t<Sshape>>
MATHPRIM_PRIMFUNC basic_view<T, Sshape, Sstride, Device> view(T *data, const Sshape &shape) {
  return basic_view<T, Sshape, Sstride, Device>(data, shape);
}

template <typename Device = device::cpu, typename T, typename Sshape, typename Sstride = default_stride_t<Sshape>>
MATHPRIM_PRIMFUNC basic_view<T, Sshape, Sstride, Device> view(T *data, const Sshape &shape, const Sstride &stride) {
  return basic_view<T, Sshape, Sstride, Device>(data, shape, stride);
}

///////////////////////////////////////////////////////////////////////////////
/// Memcpy between continuous views.
///////////////////////////////////////////////////////////////////////////////
template <typename T1, typename Sshape1, typename Sstride1, typename Dev1, typename T2, typename Sshape2,
          typename Sstride2, typename Dev2>
void copy(basic_view<T1, Sshape1, Sstride1, Dev1> dst, basic_view<T2, Sshape2, Sstride2, Dev2> src,
          bool enforce_same_shape = true) {
  MATHPRIM_INTERNAL_CHECK_THROW(src.is_contiguous() && dst.is_contiguous(), std::runtime_error,
                                "The source or destination view is not contiguous.");
  const auto total = src.numel() * sizeof(T1);

  if (enforce_same_shape) {
    MATHPRIM_INTERNAL_CHECK_THROW(src.shape() == dst.shape(), std::runtime_error,
                                  "The source and destination view must have the same shape.");
  } else {
    const auto avail = dst.numel() * sizeof(T2);
    MATHPRIM_INTERNAL_CHECK_THROW(total <= avail, std::runtime_error,
                                  "The source view is too large for the destination view.");
  }
  device::basic_memcpy<Dev2, Dev1>{}(dst.data(), src.data(), total);
}

}  // namespace mathprim
