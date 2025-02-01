#pragma once
#include <type_traits>
#include <utility>

#include "mathprim/core/defines.hpp"

namespace mathprim {

// static and dynamic
template <index_t... args>
struct index_pack;
// Use std::integer_sequence to store a sequence of indices.
template <index_t... args>
using index_seq = std::integer_sequence<index_t, args...>;
template <index_t ndim>
using make_index_seq = std::make_integer_sequence<index_t, ndim>;

///////////////////////////////////////////////////////////////////////////////
/// Tiny meta programming library.
///////////////////////////////////////////////////////////////////////////////
namespace god {

template <typename seq>
struct car;
template <index_t front, index_t... args>
struct car<index_seq<front, args...>> {
  static constexpr index_t value = front;
};

template <typename seq>
struct cdr;
template <index_t front, index_t... args>
struct cdr<index_seq<front, args...>> {
  using type = index_seq<args...>;
};

template <index_t, typename seq>
struct prepend;
template <index_t value, index_t... args>
struct prepend<value, index_seq<args...>> {
  using type = index_seq<value, args...>;
};

template <index_t value, typename seq>
struct remove {
  static_assert(value >= 0, "The value must be greater than or equal to 0.");
};
template <index_t value>
struct remove<value, index_seq<>> {
  static_assert(value == 0, "Trying to remove a non-existing value.");
};
template <index_t value, index_t front, index_t... args>
struct remove<value, index_seq<front, args...>> {
  using type = typename prepend<front, typename remove<value - 1, index_seq<args...>>::type>::type;
};
template <index_t front, index_t... args>
struct remove<0, index_seq<front, args...>> {
  using type = index_seq<args...>;
};
template <>
struct remove<0, index_seq<>> {
  using type = index_seq<>;
};

template <index_t cnt, index_t value>
struct duplicate;
template <index_t value>
struct duplicate<0, value> {
  using type = index_seq<>;
};
template <index_t cnt, index_t value>
struct duplicate {
  static_assert(cnt > 0, "The count must be greater than 0.");
  using type = typename prepend<value, typename duplicate<cnt - 1, value>::type>::type;
};

template <typename seq, index_t n>
struct get;
template <typename seq>
struct get<seq, 0> {
  static constexpr index_t value = car<seq>::value;
};
template <typename seq, index_t n>
struct get {
  static constexpr index_t value = get<typename cdr<seq>::type, n - 1>::value;
};

template <template <index_t...> class instanciation, typename seq>
struct apply_seq;
template <template <index_t...> class instanciation, index_t... args>
struct apply_seq<instanciation, index_seq<args...>> {
  using type = instanciation<args...>;
};

template <typename seq>
struct numel;
template <index_t front>
struct numel<index_seq<front>> {
  static constexpr index_t value = front;
};
template <index_t front, index_t... args>
struct numel<index_seq<front, args...>> {
  static constexpr index_t value_last = numel<index_seq<args...>>::value;
  static constexpr index_t value = (front < 0 || value_last < 0) ? -1 : front * value_last;
};

template <index_t lhs, index_t rhs>
struct arithmetic_op {
  static constexpr index_t plus = (lhs > 0 && rhs > 0) ? lhs + rhs : keep_dim;
  static constexpr index_t minus = (lhs > 0 && rhs > 0) ? lhs - rhs : keep_dim;
  static constexpr index_t multiply = (lhs > 0 && rhs > 0) ? lhs * rhs : keep_dim;
  static constexpr index_t divide = (lhs > 0 && rhs > 0) ? lhs / rhs : keep_dim;
};

template <typename lhs_seq, typename rhs_seq>
struct arithmetic_seq;  // Forward declaration.
template <index_t... lhs, index_t... rhs>
struct arithmetic_seq<index_seq<lhs...>, index_seq<rhs...>> {
  using plus = index_seq<arithmetic_op<lhs, rhs>::plus...>;
  using minus = index_seq<arithmetic_op<lhs, rhs>::minus...>;
  using multiply = index_seq<arithmetic_op<lhs, rhs>::multiply...>;
  using divide = index_seq<arithmetic_op<lhs, rhs>::divide...>;
};

/// Aliases:
template <typename seq>
constexpr index_t car_v = car<seq>::value;
template <typename seq>
using cdr_t = typename cdr<seq>::type;
template <typename seq, index_t n>
constexpr index_t get_v = get<seq, n>::value;
template <typename seq>
constexpr index_t numel_v = numel<seq>::value;
template <template <index_t...> class instanciation, typename seq>
using apply_seq_t = typename apply_seq<instanciation, seq>::type;
template <index_t value, typename seq>
using prepend_t = typename prepend<value, seq>::type;
template <index_t cnt, index_t value>
using duplicate_t = typename duplicate<cnt, value>::type;
template <index_t value, typename seq>
using remove_t = typename remove<value, seq>::type;
template <typename seq>
using to_pack = apply_seq_t<index_pack, seq>;
template <typename lhs, typename rhs>
using arithmetic_seq_plus
    = apply_seq_t<index_pack, typename arithmetic_seq<typename lhs::seq, typename rhs::seq>::plus>;
template <typename lhs, typename rhs>
using arithmetic_seq_multiply
    = apply_seq_t<index_pack, typename arithmetic_seq<typename lhs::seq, typename rhs::seq>::multiply>;

}  // namespace god

template <typename seq>
using index_seq_t = god::apply_seq_t<index_pack, seq>;

///////////////////////////////////////////////////////////////////////////////
/// wrapper for a small array, fully dynamic
///////////////////////////////////////////////////////////////////////////////
template <index_t ndim>
struct index_array {
  index_t data_[ndim] = {0};

  template <typename... Integers,
            typename = std::enable_if_t<(std::is_integral_v<Integers> && ...) && sizeof...(Integers) == ndim>>
  MATHPRIM_PRIMFUNC explicit index_array(Integers... values) noexcept : data_{static_cast<index_t>(values)...} {}

  MATHPRIM_PRIMFUNC index_array() noexcept = default;
  MATHPRIM_PRIMFUNC index_array(const index_array &) noexcept = default;
  MATHPRIM_PRIMFUNC index_array(index_array &&) noexcept = default;
  MATHPRIM_PRIMFUNC index_array &operator=(const index_array &) noexcept = default;
  MATHPRIM_PRIMFUNC index_array &operator=(index_array &&) noexcept = default;

  MATHPRIM_PRIMFUNC index_t &operator[](index_t i) noexcept {
    return data_[i];
  }

  MATHPRIM_PRIMFUNC const index_t &operator[](index_t i) const noexcept {
    return data_[i];
  }

  template <index_t i>
  MATHPRIM_PRIMFUNC index_t get() const noexcept {
    static_assert(i < ndim, "Index out of range.");
    return data_[i];
  }
};

// Iterators for index_array and index_pack
namespace internal {
template <index_t ndim>
struct index_iterator {
  index_array<ndim> current_;
  const index_array<ndim> shape_;

  // Constructor
  constexpr MATHPRIM_PRIMFUNC explicit index_iterator(const index_array<ndim> &shape, bool is_end = false) noexcept :
      shape_(shape) {
    if (is_end) {
      current_[0] = shape[0];
      for (index_t i = 1; i < ndim; ++i)
        current_[i] = 0;
    } else {
      for (index_t i = 0; i < ndim; ++i)
        current_[i] = 0;
    }
  }

  constexpr MATHPRIM_PRIMFUNC index_iterator &operator++() noexcept {
    for (index_t i = ndim - 1; i > 0; --i) {
      if (++current_[i] < shape_[i])
        return *this;
      current_[i] = 0;
    }
    ++current_[0];
    return *this;
  }

  constexpr MATHPRIM_PRIMFUNC index_iterator operator++(int) noexcept {
    index_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  constexpr MATHPRIM_PRIMFUNC const index_array<ndim> &operator*() const noexcept {
    return current_;
  }

  constexpr MATHPRIM_PRIMFUNC bool operator==(const index_iterator &other) const noexcept {
    for (index_t i = 0; i < ndim; ++i) {
      if (current_[i] != other.current_[i])
        return false;
      if (shape_[i] != other.shape_[i])
        return false;
    }
    return true;
  }

  constexpr MATHPRIM_PRIMFUNC bool operator!=(const index_iterator &other) const noexcept {
    return !(*this == other);
  }
};
}  // namespace internal

///////////////////////////////////////////////////////////////////////////////
/// dynamic and static implementation
///////////////////////////////////////////////////////////////////////////////
namespace internal {
// Static or Dynamic Router
template <index_t svalue, bool IsKeepDim = (svalue == keep_dim)>
struct router;

template <index_t svalue>
struct router<svalue, false> {
  static_assert(svalue > 0, "Cannot assign a negative value to a static index.");

  static MATHPRIM_PRIMFUNC index_t assign(index_t dvalue) noexcept {
    MATHPRIM_ASSERT(dvalue == svalue || dvalue == keep_dim);
    MATHPRIM_UNUSED(dvalue);
    return svalue;
  }
};

template <index_t svalue>
struct router<svalue, true> {
  static MATHPRIM_PRIMFUNC index_t assign(index_t dvalue) noexcept {
    MATHPRIM_ASSERT(dvalue > 0);
    return dvalue;
  }
};
}  // namespace internal

template <index_t... svalues>
struct index_pack {
  using seq = index_seq<svalues...>;
  using array_t = index_array<sizeof...(svalues)>;
  using iterator = internal::index_iterator<sizeof...(svalues)>;
  using const_iterator = internal::index_iterator<sizeof...(svalues)>;

  static_assert(((svalues > 0 || svalues == keep_dim) && ...), "Encountered an invalid value in the index_pack.");
  static constexpr index_t ndim = sizeof...(svalues);
  static constexpr bool fully_static = ((svalues > 0) && ...);

  MATHPRIM_PRIMFUNC index_pack() noexcept : dyn_(svalues...) {}

  // Constructors.
  explicit MATHPRIM_PRIMFUNC index_pack(const index_array<ndim> &array) noexcept : dyn_(array) {}

  template <typename... Integers,
            typename = std::enable_if_t<(std::is_integral_v<Integers> && ...) && sizeof...(Integers) == ndim>>
  explicit MATHPRIM_PRIMFUNC index_pack(const Integers &...args) noexcept :
      dyn_{internal::router<svalues>::assign(args)...} {}

  MATHPRIM_PRIMFUNC index_pack(const index_pack &) noexcept = default;

  MATHPRIM_PRIMFUNC index_t &operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(-ndim <= i && i < ndim && "Index out of range.");
    return i < 0 ? dyn_[ndim + i] : dyn_[i];
  }

  MATHPRIM_PRIMFUNC const index_t &operator[](index_t i) const noexcept {
    MATHPRIM_ASSERT(-ndim < i && i < ndim);
    return i < 0 ? dyn_[ndim + i] : dyn_[i];
  }

  MATHPRIM_PRIMFUNC index_t at(index_t i) const noexcept {
    return i < 0 ? dyn_[ndim + i] : dyn_[i];
  }

  template <index_t i>
  constexpr MATHPRIM_PRIMFUNC bool is_static() const noexcept {
    static_assert(-ndim <= i && i < ndim, "Index out of range.");
    return god::get_v<seq, i> > 0;
  }

  template <index_t i>
  constexpr MATHPRIM_PRIMFUNC index_t get() const noexcept {
    static_assert(-ndim <= i && i < ndim, "Index out of range.");
    constexpr index_t idx = i < 0 ? ndim + i : i;

    if constexpr (god::get_v<seq, idx> > 0) {
      return god::get_v<seq, idx>;
    } else {
      return dyn_[idx];
    }
  }

  template <index_t i>
  MATHPRIM_PRIMFUNC void set(index_t dvalue) noexcept {
    static_assert(-ndim <= i && i < ndim, "Index out of range.");
    constexpr index_t idx = i < 0 ? ndim + i : i;
    dyn_[idx] = internal::router<god::get_v<seq, idx>>::assign(dvalue);
  }

  MATHPRIM_CONSTEXPR MATHPRIM_PRIMFUNC index_t numel() const noexcept {
    if constexpr (god::numel_v<seq> > 0) {
      return god::numel_v<seq>;
    } else {
      index_t total = 1;
      for (index_t i = 0; i < ndim; ++i) {
        total *= dyn_[i];
      }
      return total;
    }
  }

  MATHPRIM_PRIMFUNC const index_array<ndim> &to_array() const noexcept {
    return dyn_;
  }

  MATHPRIM_PRIMFUNC internal::index_iterator<ndim> begin() const noexcept {
    return internal::index_iterator<ndim>{dyn_, false};
  }

  MATHPRIM_PRIMFUNC internal::index_iterator<ndim> end() const noexcept {
    return internal::index_iterator<ndim>{dyn_, true};
  }

  index_array<ndim> dyn_;
};

///////////////////////////////////////////////////////////////////////////////
/// Helpers
///////////////////////////////////////////////////////////////////////////////
template <index_t... svalues>
constexpr MATHPRIM_PRIMFUNC index_t ndim(index_pack<svalues...>) {
  return index_pack<svalues...>::ndim;
}

template <index_t... svalues>
constexpr MATHPRIM_PRIMFUNC index_t numel(index_pack<svalues...> pack) {
  return pack.numel();
}

///////////////////////////////////////////////////////////////////////////////
/// Operators
///////////////////////////////////////////////////////////////////////////////
namespace internal {

template <index_t... svalues1, index_t... svalues2, index_t... idx>
constexpr MATHPRIM_PRIMFUNC bool equal(const index_pack<svalues1...> lhs, const index_pack<svalues2...> rhs,
                                       const index_seq<idx...> & /*loop*/) {
  constexpr bool is_static_equal = ((svalues1 == svalues2 || svalues1 == keep_dim || svalues2 == keep_dim) && ...);
  return is_static_equal && ((lhs.template get<idx>() == rhs.template get<idx>()) && ...);
}

template <index_t... svalues1, index_t... svalues2, index_t... idx>
constexpr MATHPRIM_PRIMFUNC god::arithmetic_seq_plus<index_seq<svalues1...>, index_seq<svalues2...>> plus(
    const index_pack<svalues1...> &lhs, const index_pack<svalues2...> &rhs, const index_seq<idx...> & /*loop*/) {
  return {lhs.template get<idx>() + rhs.template get<idx>()...};
}

template <index_t ndim, index_t... idx>
MATHPRIM_PRIMFUNC bool equal(const index_array<ndim> &lhs, const index_array<ndim> &rhs, index_seq<idx...>) noexcept {
  return ((lhs[idx] == rhs[idx]) && ...);
}

template <index_t ndim, index_t... idx>
MATHPRIM_PRIMFUNC index_array<ndim> multiply(index_t alpha, index_array<ndim> array, index_seq<idx...>) noexcept {
  return index_array<ndim>{alpha * array[idx]...};
}

}  // namespace internal

template <index_t... svalues, index_t... svalues2>
MATHPRIM_PRIMFUNC bool operator==(const index_pack<svalues...> &lhs, const index_pack<svalues2...> &rhs) noexcept {
  return internal::equal(lhs, rhs, make_index_seq<sizeof...(svalues)>{});
}

template <index_t ndim>
MATHPRIM_PRIMFUNC bool operator==(const index_array<ndim> &lhs, const index_array<ndim> &rhs) noexcept {
  return internal::equal(lhs, rhs, make_index_seq<ndim>{});
}

template <index_t ndim>
MATHPRIM_PRIMFUNC index_array<ndim> operator*(index_t alpha, const index_array<ndim> &array) noexcept {
  return internal::multiply(alpha, array, make_index_seq<ndim>{});
}

template <index_t ndim>
MATHPRIM_PRIMFUNC index_array<ndim> operator*(const index_array<ndim> &array, index_t alpha) noexcept {
  return internal::multiply(alpha, array, make_index_seq<ndim>{});
}

template <index_t... svalues1, index_t... svalues2>
constexpr MATHPRIM_PRIMFUNC bool operator!=(const index_pack<svalues1...> &lhs,
                                            const index_pack<svalues2...> &rhs) noexcept {
  return !(lhs == rhs);
}

template <index_t... svalues1, index_t... svalues2>
constexpr MATHPRIM_PRIMFUNC god::arithmetic_seq_plus<index_seq<svalues1...>, index_seq<svalues2...>> operator+(
    const index_pack<svalues1...> &lhs, const index_pack<svalues2...> &rhs) noexcept {
  return index_pack<svalues1...>(internal::plus(lhs, rhs, make_index_seq<sizeof...(svalues1)>{}));
}

}  // namespace mathprim

///////////////////////////////////////////////////////////////////////////////
/// Structure bindings
///////////////////////////////////////////////////////////////////////////////
namespace std {

template <::mathprim::index_t ndim>
struct tuple_size<::mathprim::index_array<ndim>> : std::integral_constant<size_t, static_cast<size_t>(ndim)> {};

template <size_t i, ::mathprim::index_t ndim>
struct tuple_element<i, ::mathprim::index_array<ndim>> {
  using type = ::mathprim::index_t;
};

template <::mathprim::index_t... svalues>
struct tuple_size<::mathprim::index_pack<svalues...>> : std::integral_constant<size_t, sizeof...(svalues)> {};

template <size_t i, ::mathprim::index_t... svalues>
struct tuple_element<i, ::mathprim::index_pack<svalues...>> {
  using type = ::mathprim::index_t;
};

}  // namespace std
