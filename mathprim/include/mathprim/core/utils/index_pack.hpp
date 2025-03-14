#pragma once
#include <type_traits>
#include <utility>

#include "mathprim/core/defines.hpp"

namespace mathprim {

// static and dynamic
template <index_t... Args>
struct index_pack;

template <typename T>
struct is_index_pack : std::false_type {};
template <index_t... Args>
struct is_index_pack<index_pack<Args...>> : std::true_type {};

template <typename T>
constexpr bool is_index_pack_v = is_index_pack<T>::value;

///////////////////////////////////////////////////////////////////////////////
/// Tiny meta programming library.
///////////////////////////////////////////////////////////////////////////////
namespace god {

template <typename Seq>
struct car;
template <index_t Front, index_t... Args>
struct car<index_seq<Front, Args...>> {
  static constexpr index_t value = Front;
};

template <typename Seq>
struct cdr {
  static_assert(!std::is_same_v<Seq, Seq>, "Unsupported.");
};
template <index_t Front, index_t... Args>
struct cdr<index_seq<Front, Args...>> {
  using type = index_seq<Args...>;
};

template <typename Seq>
struct last;
template <index_t Front>
struct last<index_seq<Front>> {
  static constexpr index_t value = Front;
};
template <index_t Front, index_t... Args>
struct last<index_seq<Front, Args...>> {
  static constexpr index_t value = last<index_seq<Args...>>::value;
};

template <index_t Cnt, typename Seq>
struct drop_front;
template <index_t Cnt, index_t Front, index_t... Args>
struct drop_front<Cnt, index_seq<Front, Args...>> {
  using type = typename drop_front<Cnt - 1, index_seq<Args...>>::type;
};
template <index_t Cnt>
struct drop_front<Cnt, index_seq<>> {};
template <>
struct drop_front<0, index_seq<>> {
  using type = index_seq<>;
};
template <index_t Front, index_t... Args>
struct drop_front<0, index_seq<Front, Args...>> {
  using type = index_seq<Front, Args...>;
};

template <index_t, typename Seq>
struct prepend;
template <index_t Value, index_t... Args>
struct prepend<Value, index_seq<Args...>> {
  using type = index_seq<Value, Args...>;
};

template <index_t, typename Seq>
struct append;
template <index_t Value, index_t... Args>
struct append<Value, index_seq<Args...>> {
  using type = index_seq<Args..., Value>;
};

template <index_t Value, typename Seq>
struct remove {
  static_assert(Value >= 0, "The value must be greater than or equal to 0.");
};
template <index_t Value>
struct remove<Value, index_seq<>> {
  static_assert(Value == 0, "Trying to remove a non-existing value.");
};
template <index_t Value, index_t Front, index_t... Args>
struct remove<Value, index_seq<Front, Args...>> {
  using type = typename prepend<Front, typename remove<Value - 1, index_seq<Args...>>::type>::type;
};
template <index_t Front, index_t... Args>
struct remove<0, index_seq<Front, Args...>> {
  using type = index_seq<Args...>;
};
template <>
struct remove<0, index_seq<>> {
  using type = index_seq<>;
};

template <index_t Cnt, index_t Value>
struct duplicate;
template <index_t Value>
struct duplicate<0, Value> {
  using type = index_seq<>;
};
template <index_t Cnt, index_t Value>
struct duplicate {
  static_assert(Cnt > 0, "The count must be greater than 0.");
  using type = typename prepend<Value, typename duplicate<Cnt - 1, Value>::type>::type;
};

template <typename Seq, index_t N>
struct get;
template <typename Seq>
struct get<Seq, 0> {
  static constexpr index_t value = car<Seq>::value;
};
template <typename Seq, index_t N>
struct get {
  static constexpr index_t value = get<typename cdr<Seq>::type, N - 1>::value;
};

template <template <index_t...> class Instanciation, typename Seq>
struct apply_seq;
template <template <index_t...> class Instanciation, index_t... Args>
struct apply_seq<Instanciation, index_seq<Args...>> {
  using type = Instanciation<Args...>;
};

template <template <index_t> class Transform, typename Seq>
struct apply_elem;
template <template <index_t> class Transform, index_t... Args>
struct apply_elem<Transform, index_seq<Args...>> {
  using type = index_seq<Transform<Args>::value...>;
};

template <typename Seq>
struct numel;
template <index_t Front>
struct numel<index_seq<Front>> {
  static constexpr index_t value = Front;
};
template <index_t Front, index_t... Args>
struct numel<index_seq<Front, Args...>> {
  static constexpr index_t value_last = numel<index_seq<Args...>>::value;
  static constexpr index_t value = (Front < 0 || value_last < 0) ? -1 : Front * value_last;
};

template <index_t Lhs, index_t Rhs>
struct arithmetic_op {
  static constexpr index_t plus = (Lhs > 0 && Rhs > 0) ? Lhs + Rhs : keep_dim;
  static constexpr index_t minus = (Lhs > 0 && Rhs > 0) ? Lhs - Rhs : keep_dim;
  static constexpr index_t multiply = (Lhs > 0 && Rhs > 0) ? Lhs * Rhs : keep_dim;
  static constexpr index_t divide = (Lhs > 0 && Rhs > 0) ? Lhs / Rhs : keep_dim;
  static constexpr index_t up_div = (Lhs > 0 && Rhs > 0) ? (Lhs + Rhs - 1) / Rhs : keep_dim;
};

template <typename LhsSeq, typename RhsSeq>
struct arithmetic_seq;  // Forward declaration.
template <index_t... Lhs, index_t... Rhs>
struct arithmetic_seq<index_seq<Lhs...>, index_seq<Rhs...>> {
  using plus = index_seq<arithmetic_op<Lhs, Rhs>::plus...>;
  using minus = index_seq<arithmetic_op<Lhs, Rhs>::minus...>;
  using multiply = index_seq<arithmetic_op<Lhs, Rhs>::multiply...>;
  using divide = index_seq<arithmetic_op<Lhs, Rhs>::divide...>;
  using up_div = index_seq<arithmetic_op<Lhs, Rhs>::up_div...>;
};

/// Aliases:
template <typename Seq>
constexpr index_t car_v = car<Seq>::value;
template <typename Seq>
constexpr index_t last_v = last<Seq>::value;
template <typename Seq>
using cdr_t = typename cdr<Seq>::type;
template <typename Seq, index_t N>
constexpr index_t get_v = get<Seq, N>::value;
template <typename Seq>
constexpr index_t numel_v = numel<Seq>::value;
template <template <index_t...> class Instanciation, typename Seq>
using apply_seq_t = typename apply_seq<Instanciation, Seq>::type;
template <template <index_t> class Transform, typename Seq>
using apply_elem_t = typename apply_elem<Transform, Seq>::type;
template <index_t Cnt, typename Seq>
using drop_front_t = typename drop_front<Cnt, Seq>::type;
template <index_t Value, typename Seq>
using prepend_t = typename prepend<Value, Seq>::type;
template <index_t Cnt, index_t Value>
using duplicate_t = typename duplicate<Cnt, Value>::type;
template <index_t Value, typename Seq>
using remove_t = typename remove<Value, Seq>::type;
template <typename Seq>
using to_pack = apply_seq_t<index_pack, Seq>;
template <typename Lhs, typename Rhs>
using arithmetic_seq_plus
    = apply_seq_t<index_pack, typename arithmetic_seq<typename Lhs::seq, typename Rhs::seq>::plus>;
template <typename Lhs, typename Rhs>
using arithmetic_seq_multiply
    = apply_seq_t<index_pack, typename arithmetic_seq<typename Lhs::seq, typename Rhs::seq>::multiply>;
template <typename Lhs, typename Rhs>
using arithmetic_seq_up_div
    = apply_seq_t<index_pack, typename arithmetic_seq<typename Lhs::seq, typename Rhs::seq>::up_div>;
template <index_t Value, typename Seq>
using append_t = typename append<Value, Seq>::type;

}  // namespace god

template <typename Seq>
using index_seq_t = god::apply_seq_t<index_pack, Seq>;

///////////////////////////////////////////////////////////////////////////////
/// wrapper for a small array, fully dynamic
///////////////////////////////////////////////////////////////////////////////
template <index_t Ndim>
struct index_array {
  index_t data_[Ndim] = {0};

  template <typename... Integers,
            typename = std::enable_if_t<(std::is_integral_v<Integers> && ...) && sizeof...(Integers) == Ndim>>
  MATHPRIM_PRIMFUNC explicit index_array(Integers... values) noexcept : data_{static_cast<index_t>(values)...} {}

  MATHPRIM_PRIMFUNC explicit index_array(const index_t *data) noexcept {
    for (index_t i = 0; i < Ndim; ++i) {
      data_[i] = data[i];
    }
  }
  index_array() noexcept = default;
  index_array(const index_array &) noexcept = default;
  index_array(index_array &&) noexcept = default;
  index_array &operator=(const index_array &) noexcept = default;
  index_array &operator=(index_array &&) noexcept = default;

  MATHPRIM_PRIMFUNC index_t &operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(0 <= i && i < Ndim);
    return data_[i];
  }

  MATHPRIM_PRIMFUNC const index_t &operator[](index_t i) const noexcept {
    MATHPRIM_ASSERT(0 <= i && i < Ndim);
    return data_[i];
  }

  template <index_t I>
  MATHPRIM_PRIMFUNC index_t get() const noexcept {
    static_assert(I < Ndim, "Index out of range.");
    return data_[I];
  }

  MATHPRIM_PRIMFUNC static index_array<Ndim> constant(index_t value) noexcept {
    index_array<Ndim> result;
    for (index_t i = 0; i < Ndim; ++i) {
      result[i] = value;
    }
    return result;
  }
};

template <>
struct index_array<0> {
  index_array() noexcept = default;
  index_array(const index_array &) noexcept = default;
  index_array(index_array &&) noexcept = default;
  index_array &operator=(const index_array &) noexcept = default;
  index_array &operator=(index_array &&) noexcept = default;
  MATHPRIM_PRIMFUNC index_t &operator[](index_t) noexcept;
  MATHPRIM_PRIMFUNC const index_t &operator[](index_t i) const noexcept;
  template <index_t I>
  MATHPRIM_PRIMFUNC index_t get() const noexcept;
  MATHPRIM_PRIMFUNC static index_array<0> constant(index_t) noexcept;
};

template <>
struct index_array<1> {
  index_t data_[1] = {0};

  MATHPRIM_PRIMFUNC explicit index_array(index_t value) noexcept : data_{value} {}

  index_array() noexcept = default;
  index_array(const index_array &) noexcept = default;
  index_array(index_array &&) noexcept = default;
  index_array &operator=(const index_array &) noexcept = default;
  index_array &operator=(index_array &&) noexcept = default;

  MATHPRIM_PRIMFUNC operator index_t() const noexcept {  // NOLINT: google-explicit-constructor
    return data_[0];
  }

  MATHPRIM_PRIMFUNC index_t &operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(i == 0);
    return data_[i];
  }

  MATHPRIM_PRIMFUNC const index_t &operator[](index_t i) const noexcept {
    MATHPRIM_ASSERT(i == 0);
    return data_[i];
  }

  template <index_t I>
  MATHPRIM_PRIMFUNC index_t get() const noexcept {
    static_assert(I == 0, "Index out of range.");
    return data_[0];
  }
  MATHPRIM_PRIMFUNC static index_array<1> constant(index_t value) noexcept { return index_array<1>{value}; }
};

template <index_t Ndim>
MATHPRIM_PRIMFUNC index_array<Ndim> operator*(const index_array<Ndim> &lhs, const index_array<Ndim> &rhs) noexcept {
  index_array<Ndim> result;
  for (index_t i = 0; i < Ndim; ++i) {
    result[i] = lhs[i] * rhs[i];
  }
  return result;
}

template <index_t Ndim>
MATHPRIM_PRIMFUNC index_array<Ndim> operator+(const index_array<Ndim> &lhs, const index_array<Ndim> &rhs) noexcept {
  index_array<Ndim> result;
  for (index_t i = 0; i < Ndim; ++i) {
    result[i] = lhs[i] + rhs[i];
  }
  return result;
}

template <index_t Ndim>
MATHPRIM_PRIMFUNC index_array<Ndim> up_div(const index_array<Ndim> &array, const index_array<Ndim> &divisor) noexcept {
  index_array<Ndim> result;
  for (index_t i = 0; i < Ndim; ++i) {
    MATHPRIM_ASSERT(divisor[i] > 0 && "Divisor must be greater than 0.");
    result[i] = (array[i] + divisor[i] - 1) / divisor[i];
  }
  return result;
}

// Iterators for index_array and index_pack
namespace internal {
template <index_t Ndim>
struct index_iterator {
  index_array<Ndim> current_;
  const index_array<Ndim> shape_;

  // Constructor
  constexpr MATHPRIM_PRIMFUNC explicit index_iterator(const index_array<Ndim> &shape, bool is_end = false) noexcept :
      shape_(shape) {
    if (is_end) {
      current_[0] = shape[0];
      for (index_t i = 1; i < Ndim; ++i)
        current_[i] = 0;
    } else {
      for (index_t i = 0; i < Ndim; ++i)
        current_[i] = 0;
    }
  }

  constexpr MATHPRIM_PRIMFUNC index_iterator &operator++() noexcept {
    for (index_t i = Ndim - 1; i > 0; --i) {
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

  constexpr MATHPRIM_PRIMFUNC const index_array<Ndim> &operator*() const noexcept { return current_; }

  constexpr MATHPRIM_PRIMFUNC bool operator==(const index_iterator &other) const noexcept {
    for (index_t i = 0; i < Ndim; ++i) {
      if (current_[i] != other.current_[i])
        return false;
      if (shape_[i] != other.shape_[i])
        return false;
    }
    return true;
  }

  constexpr MATHPRIM_PRIMFUNC bool operator!=(const index_iterator &other) const noexcept { return !(*this == other); }
};

template <typename T>
MATHPRIM_PRIMFUNC void myswap(T &lhs, T &rhs) noexcept {
#ifdef __CUDA_ARCH__
  T tmp = lhs;
  lhs = rhs;
  rhs = tmp;
#else
  T tmp = std::move(lhs);
  lhs = std::move(rhs);
  rhs = std::move(tmp);
#endif
}

template <index_t Ndim>
void swap_impl(index_array<Ndim> &lhs, index_array<Ndim> &rhs) noexcept {
  for (index_t i = 0; i < Ndim; ++i) {
    myswap<index_t>(lhs[i], rhs[i]);
  }
}

}  // namespace internal

template <index_t NDim>
MATHPRIM_PRIMFUNC internal::index_iterator<NDim> begin(const index_array<NDim> &shape) noexcept {
  return internal::index_iterator<NDim>{shape};
}

template <index_t NDim>
MATHPRIM_PRIMFUNC internal::index_iterator<NDim> end(const index_array<NDim> &shape) noexcept {
  return internal::index_iterator<NDim>{shape, true};
}

///////////////////////////////////////////////////////////////////////////////
/// dynamic and static implementation
///////////////////////////////////////////////////////////////////////////////
namespace internal {
// Static or Dynamic Router
template <index_t Svalue, bool IsKeepDim = (Svalue == keep_dim)>
struct router;

template <index_t Svalue>
struct router<Svalue, false> {
  static_assert(Svalue > 0, "Cannot assign a negative value to a static index.");

  static MATHPRIM_PRIMFUNC index_t assign(index_t dvalue) noexcept {
    MATHPRIM_ASSERT(dvalue == Svalue || dvalue == keep_dim);
    MATHPRIM_UNUSED(dvalue);
    return Svalue;
  }
};

template <index_t Svalue>
struct router<Svalue, true> {
  static MATHPRIM_PRIMFUNC index_t assign(index_t dvalue) noexcept {
    MATHPRIM_ASSERT(dvalue >= 0);
    return dvalue;
  }
};

template <index_t Ndim>
struct make_index_seq_impl {
  static_assert(Ndim > 0, "The dimension must be greater than 0.");
  using type = god::append_t<Ndim - 1, typename make_index_seq_impl<Ndim - 1>::type>;
};

template <>
struct make_index_seq_impl<0> {
  using type = index_seq<>;
};

template <index_t Ndim>
using make_index_seq = typename make_index_seq_impl<Ndim>::type;

}  // namespace internal

template <index_t Ndim>
using make_index_seq = internal::make_index_seq<Ndim>;

template <index_t... Svalues>
struct index_pack {
  using seq = index_seq<Svalues...>;
  using arr = index_array<sizeof...(Svalues)>;
  using iterator = internal::index_iterator<sizeof...(Svalues)>;
  using const_iterator = internal::index_iterator<sizeof...(Svalues)>;

  static_assert(((Svalues > 0 || Svalues == keep_dim) && ...), "Encountered an invalid value in the index_pack.");
  static constexpr index_t ndim = sizeof...(Svalues);
  static constexpr bool fully_static = ((Svalues > 0) && ...);

  MATHPRIM_PRIMFUNC index_pack() noexcept : dyn_(Svalues...) {}

  // Constructors.
  explicit MATHPRIM_PRIMFUNC index_pack(const index_array<ndim> &array) noexcept : dyn_(array) {}

  template <typename... Integers,
            typename = std::enable_if_t<(std::is_integral_v<Integers> && ...) && sizeof...(Integers) == ndim>>
  explicit MATHPRIM_PRIMFUNC index_pack(const Integers &...args) noexcept :
      dyn_{internal::router<Svalues>::assign(static_cast<index_t>(args))...} {}

  index_pack(const index_pack &) noexcept = default;
  index_pack(index_pack &&) noexcept = default;
  index_pack &operator=(const index_pack &) noexcept = default;
  index_pack &operator=(index_pack &&) noexcept = default;

  MATHPRIM_PRIMFUNC index_t &operator[](index_t i) noexcept {
    MATHPRIM_ASSERT(-ndim <= i && i < ndim && "Index out of range.");
    return i < 0 ? dyn_[ndim + i] : dyn_[i];
  }

  MATHPRIM_PRIMFUNC const index_t &operator[](index_t i) const noexcept {
    MATHPRIM_ASSERT(-ndim < i && i < ndim);
    return i < 0 ? dyn_[ndim + i] : dyn_[i];
  }

  MATHPRIM_PRIMFUNC index_t at(index_t i) const noexcept { return i < 0 ? dyn_[ndim + i] : dyn_[i]; }

  template <index_t I>
  constexpr MATHPRIM_PRIMFUNC bool is_static() const noexcept {
    static_assert(-ndim <= I && I < ndim, "Index out of range.");
    return god::get_v<seq, I> > 0;
  }

  template <index_t I>
  constexpr MATHPRIM_PRIMFUNC index_t get() const noexcept {
    static_assert(-ndim <= I && I < ndim, "Index out of range.");
    constexpr index_t idx = I < 0 ? ndim + I : I;

    if constexpr (god::get_v<seq, idx> > 0) {
      return god::get_v<seq, idx>;
    } else {
      return dyn_[idx];
    }
  }

  template <index_t I>
  MATHPRIM_PRIMFUNC void set(index_t dvalue) noexcept {
    static_assert(-ndim <= I && I < ndim, "Index out of range.");
    constexpr index_t idx = I < 0 ? ndim + I : I;
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

  MATHPRIM_PRIMFUNC const index_array<ndim> &to_array() const noexcept { return dyn_; }

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
template <index_t... Svalues>
constexpr MATHPRIM_PRIMFUNC index_t ndim(index_pack<Svalues...>) {
  return index_pack<Svalues...>::ndim;
}

template <index_t... Svalues>
constexpr MATHPRIM_PRIMFUNC index_t numel(index_pack<Svalues...> pack) {
  return pack.numel();
}

///////////////////////////////////////////////////////////////////////////////
/// Operators
///////////////////////////////////////////////////////////////////////////////
namespace internal {
template <typename Lhs, typename Rhs, bool DimEqual = Lhs::ndim == Rhs::ndim>
struct compile_time_equal : std::false_type {};

template <index_t... Svalues1, index_t... Svalues2>
struct compile_time_equal<index_pack<Svalues1...>, index_pack<Svalues2...>, true> {
  static constexpr bool value = ((Svalues1 == Svalues2 && Svalues1 != keep_dim && Svalues2 != keep_dim) && ...);
};
template <typename Lhs, typename Rhs, bool DimEqual = Lhs::ndim == Rhs::ndim>
struct compile_time_capable : std::false_type {};

template <index_t... Svalues1, index_t... Svalues2>
struct compile_time_capable<index_pack<Svalues1...>, index_pack<Svalues2...>, true> {
  static constexpr bool value = ((Svalues1 == Svalues2 || Svalues1 == keep_dim || Svalues2 == keep_dim) && ...);
};

template <typename Lhs, typename Rhs>
constexpr bool is_compile_time_equal_v = compile_time_equal<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
constexpr bool is_compile_time_capable_v = compile_time_capable<Lhs, Rhs>::value;

template <index_t... Svalues1, index_t... Svalues2, index_t... Idx>
constexpr MATHPRIM_PRIMFUNC bool equal(const index_pack<Svalues1...> lhs, const index_pack<Svalues2...> rhs,
                                       const index_seq<Idx...> & /*loop*/) {
  MATHPRIM_UNUSED(lhs);
  MATHPRIM_UNUSED(rhs);
  constexpr bool is_static_equal = ((Svalues1 == Svalues2 || Svalues1 == keep_dim || Svalues2 == keep_dim) && ...);
  return is_static_equal && ((lhs.template get<Idx>() == rhs.template get<Idx>()) && ...);
}

template <index_t... Svalues1, index_t... Svalues2, index_t... Idx>
constexpr MATHPRIM_PRIMFUNC god::arithmetic_seq_plus<index_seq<Svalues1...>, index_seq<Svalues2...>> plus(
    const index_pack<Svalues1...> &lhs, const index_pack<Svalues2...> &rhs, const index_seq<Idx...> & /*loop*/) {
  return {lhs.template get<Idx>() + rhs.template get<Idx>()...};
}

template <index_t Ndim, index_t... Idx>
MATHPRIM_PRIMFUNC bool equal(const index_array<Ndim> &lhs, const index_array<Ndim> &rhs, index_seq<Idx...>) noexcept {
  return ((lhs[Idx] == rhs[Idx]) && ...);
}

template <index_t Ndim, index_t... Idx>
MATHPRIM_PRIMFUNC index_array<Ndim> multiply(index_t alpha, index_array<Ndim> array, index_seq<Idx...>) noexcept {
  return index_array<Ndim>{alpha * array[Idx]...};
}

// Cast from one index_pack to another.
template <typename To, index_t... Idx, typename From>
MATHPRIM_PRIMFUNC To safe_cast_impl(const From &from, index_seq<Idx...>) noexcept {
  return To{from.template get<Idx>()...};
}

template <typename To, typename From, typename = std::enable_if_t<is_compile_time_capable_v<From, To>>>
MATHPRIM_PRIMFUNC To safe_cast(const From &from) noexcept {
  return safe_cast_impl<To>(from, make_index_seq<To::ndim>{});
}

template <typename From, typename To>
struct is_castable;
template <index_t... FromValues, index_t... ToValues>
struct is_castable<index_pack<FromValues...>, index_pack<ToValues...>> {
  static constexpr bool value = ((FromValues == ToValues || FromValues == keep_dim || ToValues == keep_dim) && ...);
};
template <typename From, typename To>
static constexpr bool is_castable_v = is_castable<From, To>::value;
}  // namespace internal

template <index_t... Svalues, index_t... Svalues2>
MATHPRIM_PRIMFUNC bool operator==(const index_pack<Svalues...> &lhs, const index_pack<Svalues2...> &rhs) noexcept {
  return internal::equal(lhs, rhs, make_index_seq<sizeof...(Svalues)>{});
}

template <index_t Ndim>
MATHPRIM_PRIMFUNC bool operator==(const index_array<Ndim> &lhs, const index_array<Ndim> &rhs) noexcept {
  return internal::equal(lhs, rhs, make_index_seq<Ndim>{});
}

template <index_t Ndim>
MATHPRIM_PRIMFUNC index_array<Ndim> operator*(index_t alpha, const index_array<Ndim> &array) noexcept {
  return internal::multiply(alpha, array, make_index_seq<Ndim>{});
}

template <index_t Ndim>
MATHPRIM_PRIMFUNC index_array<Ndim> operator*(const index_array<Ndim> &array, index_t alpha) noexcept {
  return internal::multiply(alpha, array, make_index_seq<Ndim>{});
}

template <index_t... Svalues1, index_t... Svalues2>
constexpr MATHPRIM_PRIMFUNC bool operator!=(const index_pack<Svalues1...> &lhs,
                                            const index_pack<Svalues2...> &rhs) noexcept {
  return !(lhs == rhs);
}

template <index_t... Svalues1, index_t... Svalues2>
constexpr MATHPRIM_PRIMFUNC god::arithmetic_seq_plus<index_pack<Svalues1...>, index_pack<Svalues2...>> operator+(
    const index_pack<Svalues1...> &lhs, const index_pack<Svalues2...> &rhs) noexcept {
  return god::arithmetic_seq_plus<index_pack<Svalues1...>, index_pack<Svalues2...>>(
      internal::plus(lhs, rhs, make_index_seq<sizeof...(Svalues1)>{}));
}

template <index_t... Svalues1, index_t... Svalues2>
constexpr MATHPRIM_PRIMFUNC god::arithmetic_seq_up_div<index_pack<Svalues1...>, index_pack<Svalues2...>> up_div(
    const index_pack<Svalues1...> &lhs, const index_pack<Svalues2...> &rhs) noexcept {
  return god::arithmetic_seq_up_div<index_pack<Svalues1...>, index_pack<Svalues2...>>(
      up_div(lhs.to_array(), rhs.to_array()));
}

}  // namespace mathprim

///////////////////////////////////////////////////////////////////////////////
/// Structure bindings
///////////////////////////////////////////////////////////////////////////////
namespace std {

template <::mathprim::index_t Ndim>
struct tuple_size<::mathprim::index_array<Ndim>> : std::integral_constant<size_t, static_cast<size_t>(Ndim)> {};

template <size_t I, ::mathprim::index_t Ndim>
struct tuple_element<I, ::mathprim::index_array<Ndim>> {
  using type = ::mathprim::index_t;
};

template <::mathprim::index_t... Svalues>
struct tuple_size<::mathprim::index_pack<Svalues...>> : std::integral_constant<size_t, sizeof...(Svalues)> {};

template <size_t I, ::mathprim::index_t... Svalues>
struct tuple_element<I, ::mathprim::index_pack<Svalues...>> {
  using type = ::mathprim::index_t;
};

}  // namespace std
