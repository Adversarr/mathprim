#pragma once
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"
#include "mathprim/core/view.hpp"
#include "mathprim/dnn/basic_module.hpp"

namespace mathprim::dnn {
namespace internal {
template <typename Front, typename Back, size_t Idx, bool IsLast>
class sequential_impl_rec
    : public basic_module<sequential_impl_rec<Front, Back, Idx, IsLast>, typename Front::scalar_t, typename Front::device_t,
                          typename Front::in_shape, typename Back::out_shape> {
public:
  using base = basic_module<sequential_impl_rec<Front, Back, Idx, IsLast>, typename Front::scalar_t,
                            typename Front::device_t, typename Front::in_shape, typename Back::out_shape>;
  friend base;
  using in_shape = typename base::in_shape;
  using out_shape = typename base::out_shape;

  using in_batch = typename base::in_batch;
  using out_batch = typename base::out_batch;
  using const_in_batch = typename base::const_in_batch;
  using const_out_batch = typename base::const_out_batch;
  using compile_return_t = typename base::compile_return_t;

  using intermediate_batch = typename Front::out_batch;
  using const_intermediate_batch = typename Front::const_out_batch;

  template <typename Blas, typename ParImpl>
  using ctx_t = typename base::template ctx_t<Blas, ParImpl>;
  using device_t = typename base::device_t;
  using scalar_t = typename base::scalar_t;

  using intermediate_shape = typename Front::out_shape;
  using intermediate_view = contiguous_view<scalar_t, intermediate_shape, device_t>;
  using intermediate_buffer = to_buffer_t<intermediate_view>;

  static_assert(std::is_same_v<scalar_t, typename Back::scalar_t>, "Scalar type must be the same.");
  static_assert(std::is_same_v<device_t, typename Back::device_t>, "Device type must be the same.");
  static_assert(std::is_same_v<intermediate_shape, typename Back::in_shape>, "Shape mismatch.");

  sequential_impl_rec() = default;
  // MATHPRIM_INTERNAL_MOVE(sequential_two, default);
  // sequential_two(Front&& front, Back&& back) : front_(std::move(front)), back_(std::move(back)) {}
  template <typename... Args>
  explicit sequential_impl_rec(Front&& front, Args&&... args) :
      front_(std::move(front)), back_(std::forward<Args>(args)...) {}

  template <typename Blas, typename ParImpl>
  compile_return_t compile_impl(ctx_t<Blas, ParImpl>& c) {
    // Front
    c.push_prefix(std::to_string(Idx));
    // Intermediate buffer
    auto [z, dldz] = front_.compile(c, base::curr_x_, base::curr_dl_dx_);
    c.pop_prefix();

    compile_return_t out;
    if constexpr (IsLast) {  // Back is not a tree, responsible to push a prefix.
      c.push_prefix(std::to_string(Idx + 1));
      out = back_.compile(c, z, dldz);
      c.pop_prefix();
    } else {
      out = back_.compile(c, z, dldz);
    }
    return out;
  }

  void reset_parameters_impl() {
    front_.reset_parameters();
    back_.reset_parameters();
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Comptues
  ///////////////////////////////////////////////////////////////////////////////
  template <typename Blas, typename ParImpl>
  out_batch forward_impl(ctx_t<Blas, ParImpl>& c) {
    front_.forward(c);
    return back_.forward(c);
  }

  template <typename Blas, typename ParImpl>
  void backward_impl(ctx_t<Blas, ParImpl>& c, bool compute_dldx) {
    back_.backward(c, true);  // computes dL/dZ is necessary for front_
    front_.backward(c, compute_dldx);
  }

  template <typename Blas, typename ParImpl>
  void zero_grad_impl(ctx_t<Blas, ParImpl>& c) {
    front_.zero_grad(c);
    back_.zero_grad(c);
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Meta data
  ///////////////////////////////////////////////////////////////////////////////
  index_t total_weights_impl() { return front_.total_weights() + back_.total_weights(); }
  in_shape input_shape_impl() const noexcept { return front_.input_shape(); }
  out_shape out_shape_impl() const noexcept { return back_.out_shape(); }
  ///////////////////////////////////////////////////////////////////////////////
  /// Accessors
  ///////////////////////////////////////////////////////////////////////////////

  const_intermediate_batch intermediate() const noexcept { return front_.output(); }
  intermediate_batch intermediate_gradient() const noexcept { return front_.output_gradient(); }

  Front& front() noexcept { return front_; }
  const Front& front() const noexcept { return front_; }
  Back& back() noexcept { return back_; }
  const Back& back() const noexcept { return back_; }

  template <index_t I>
  auto& get() {
    if constexpr (I == 0) {
      return front_;
    } else if constexpr (I == 1) {
      if constexpr (!IsLast) {
        return back_.template get<I - 1>();
      } else {
        return back_;
      }
    } else {
      return back_.template get<I - 1>();
    }
  }

private:
  Front front_;  // x -> z
  Back back_;    // z -> y
};
template <size_t Total>
struct sequential_builder {
  template <typename ... Modules>
  struct impl;

  template <typename Front, typename End>
  struct impl<Front, End> {
    using type = sequential_impl_rec<Front, End, Total - 2, true>;
  };

  template <typename Front, typename Next, typename ... Rest>
  struct impl<Front, Next, Rest...> {
    static_assert(Total >= sizeof...(Rest) + 2,
                  "[Internal] Total must be greater than or equal to the number of modules.");
    using type = sequential_impl_rec<Front, typename impl<Next, Rest...>::type, Total - sizeof...(Rest) - 2, false>;
  };

  template <typename... Modules>
  using type = typename impl<Modules...>::type;
};

template <typename ...Modules>
struct last_out_shape;
template <typename Last>
struct last_out_shape<Last> {
  using type = typename Last::out_shape;
};

template <typename Front, typename ...Rest>
struct last_out_shape<Front, Rest...> {
  using type = typename last_out_shape<Rest...>::type;
};
}  // namespace internal

template <typename Front, typename Next, typename... Rest>
using sequential = typename internal::sequential_builder<sizeof...(Rest) + 2>::template type<Front, Next, Rest...>;

}  // namespace mathprim::dnn