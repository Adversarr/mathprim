#pragma once
#include <algorithm>
#include <vector>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/core/utils/parameter_item.hpp"
#include "mathprim/core/view.hpp"

namespace mathprim::dnn {
template <typename Derived, typename Scalar, typename Device, typename InShape, typename OutShape>
class basic_module;

// Responsible for managing the memory of all dL/dW and the network.
template <typename Scalar, typename Device, typename Blas, typename ParImpl, typename InShape, typename OutShape>
struct basic_ctx {
  using gradient_view = contiguous_vector_view<Scalar, Device>;
  using gradient_buffer = contiguous_vector_buffer<Scalar, Device>;
  using parameter_t = parameter_item<Scalar, Device>;
  using parameter_value_view = typename parameter_t::view_type;
  using const_out_batch = batched<contiguous_view<const Scalar, OutShape, Device>>;
  using out_batch = batched<contiguous_view<Scalar, OutShape, Device>>;
  using in_batch = batched<contiguous_view<Scalar, InShape, Device>>;

  /// @brief Get the blas in use.
  blas::basic_blas<Blas, Scalar, Device>& blas() noexcept { return blas_; }
  par::parfor<ParImpl>& parallel() noexcept { return par_; }

  parameter_t push(parameter_value_view values, const std::string& name) {
    const auto size = values.numel();
    MATHPRIM_INTERNAL_CHECK_THROW(curr_offset_ + size <= total_weights_, std::runtime_error,
                                  "The total number of weights is not equal to the requested size.");
    auto subgrad = dL_dW_.view().sub(curr_offset_, curr_offset_ + size);
    std::string full_name;
    for (const auto& prefix : naming_prefixes_) {
      full_name += prefix;
      full_name.push_back('.');
    }
    if (name.empty()) {
      // Handle the case where name is empty
      full_name += "???";
    } else {
      full_name += name;
    }
    curr_offset_ += size;
    return weights_info_.emplace_back(values, subgrad, full_name);
  }

  void push_prefix(const std::string& prefix) {
    naming_prefixes_.push_back(prefix);
  }

  void pop_prefix() {
    MATHPRIM_INTERNAL_CHECK_THROW(!naming_prefixes_.empty(), std::runtime_error, "No prefix to pop.");
    naming_prefixes_.pop_back();
  }

  template <typename ModuleDerived>
  void compile(basic_module<ModuleDerived, Scalar, Device, InShape, OutShape>& module, index_t batch_size) {
    reset(module.total_weights());
    module.compile(*this, batch_size);
    MATHPRIM_INTERNAL_CHECK_THROW(curr_offset_ == total_weights_, std::runtime_error,
                                  "The total number of weights is not equal to the requested size.");
    MATHPRIM_INTERNAL_CHECK_THROW(naming_prefixes_.empty(), std::runtime_error, "Prefix stack is not empty.");

    using batch_shape_t = typename in_batch::shape_at_compile_time;
    batch_shape_t shape = batched_shape(batch_size, module.input_shape());
    x_ = make_buffer<Scalar, Device>(shape);
    y_ = module.output();
    dl_dy_ = module.output_gradient();
  }

  template <typename ModuleDerived>
  void zero_grad(basic_module<ModuleDerived, Scalar, Device, InShape, OutShape>& module) {
    if (dL_dW_) dL_dW_.fill_bytes(0);
    module.zero_grad(*this);
  }

  template <typename ModuleDerived>
  auto forward(basic_module<ModuleDerived, Scalar, Device, InShape, OutShape>& module) {
    return module.forward(*this, x_.const_view());
  }

  template <typename ModuleDerived>
  void backward(basic_module<ModuleDerived, Scalar, Device, InShape, OutShape>& module, in_batch dl_dx = {}) {
    module.backward(*this, dl_dx);
  }

  in_batch input() noexcept { return x_.view(); }
  in_batch input() const noexcept { return x_.const_view(); }
  const_out_batch output() const noexcept { return y_; }
  out_batch output_gradient() const noexcept { return dl_dy_; }

  gradient_view params_gradient() noexcept { return dL_dW_.view(); }
  gradient_view params_gradient() const noexcept { return dL_dW_.const_view(); }

  template <typename Fn>
  void for_each_parameter(Fn&& fn) {
    std::for_each(weights_info_.begin(), weights_info_.end(), std::forward<Fn>(fn));
  }

private:
  // Information for each weight
  void reset(index_t total_weights) {
    total_weights_ = total_weights;
    if (total_weights > 0) {
      dL_dW_ = make_buffer<Scalar, Device>(total_weights);
    }
    curr_offset_ = 0;
  }

  // during compilation:
  index_t curr_offset_;
  index_t total_weights_;
  Blas blas_;
  ParImpl par_;

  std::vector<std::string> naming_prefixes_;

  to_buffer_t<in_batch> x_; // always stores the current input
  const_out_batch y_;       // always stores the current output
  out_batch dl_dy_;         // always stores the current dL/dY

  contiguous_vector_buffer<Scalar, Device> dL_dW_;
  std::vector<parameter_t> weights_info_;
};

/**
 * @brief Base class for NNs.
 *
 * Note: all the variables produced by the module, are stored in the module.(excepts dL/dW)
 *
 */
template <typename Derived, typename Scalar, typename Device, typename InShape, typename OutShape>
class basic_module {
public:
  using in_shape = InShape;
  using out_shape = OutShape;
  using scalar_t = Scalar;
  using device_t = Device;

  using in_data_view = contiguous_view<Scalar, InShape, Device>;
  using out_data_view = contiguous_view<Scalar, OutShape, Device>;
  using const_in_data = contiguous_view<const Scalar, InShape, Device>;
  using const_out_data = contiguous_view<const Scalar, OutShape, Device>;

  using in_batch = batched<in_data_view>;
  using out_batch = batched<out_data_view>;
  using const_in_batch = batched<const_in_data>;
  using const_out_batch = batched<const_out_data>;

  template <typename Blas, typename ParImpl>
  using ctx_t = basic_ctx<Scalar, Device, Blas, ParImpl, InShape, OutShape>;

  // Fused buffer for weights and gradients
  using compile_return_t = std::pair<const_out_batch, out_batch>;

  basic_module() = default;
  MATHPRIM_INTERNAL_MOVE(basic_module, default);

  Derived& derived() noexcept { return *static_cast<Derived*>(this); }
  const Derived& derived() const noexcept { return *static_cast<const Derived*>(this); }

  /// @brief Get view to outputs.
  const_out_batch output() const noexcept { return curr_y_; }

  out_batch output_gradient() const noexcept {
    MATHPRIM_ASSERT(has_compiled_);
    return curr_dl_dy_;
  }

  /// @brief Ensure all the buffers for inputs and outputs are prepared.
  ///        After this, the module is ready to compute.
  template <typename Blas, typename ParImpl>
  compile_return_t compile(ctx_t<Blas, ParImpl>& c, index_t batch_size) {
    curr_batch_size_ = batch_size;
    // returns the view of output Y and dL/dY
    std::tie(curr_y_, curr_dl_dy_) = derived().compile_impl(c);
    has_compiled_ = true;

    return {curr_y_, curr_dl_dy_};
  }

  index_t current_batch_size() const noexcept { return curr_batch_size_; }

  template <typename Blas, typename ParImpl>
  void zero_grad(ctx_t<Blas, ParImpl>& c) {
    MATHPRIM_ASSERT(has_compiled_);
    zeros(curr_dl_dy_);
    derived().zero_grad_impl(c);
  }

  // (W, X) -> Y
  template <typename Blas, typename ParImpl>
  out_batch forward(ctx_t<Blas, ParImpl>& c, const_in_batch batched_x) {
    MATHPRIM_ASSERT(has_compiled_);
    curr_x_ = batched_x;
    return derived().forward_impl(c);
  }

  // dL/dY -> dL/dW(must), dL/dX(if set)
  template <typename Blas, typename ParImpl>
  void backward(ctx_t<Blas, ParImpl>& c, in_batch dl_dx) {
    MATHPRIM_ASSERT(has_compiled_);
    derived().backward_impl(c, dl_dx);
  }

  // // Notes: Future work
  // // dL/dY -> dL/dX
  // void backward_input(const_out_batch batched_dl_dy);
  // // dL/d(dL/dX) -> dL/dW
  // void backward_backward_input();
  // // dL/d(dL/dX) -> dL/dX
  // void backward_backward_input_input();
  // // Accumulates dL/d(dL/dX)
  // void accumulate_input_input_gradients();

  ///////// Meta information /////////
  InShape input_shape() const noexcept { return derived().input_shape_impl(); }
  OutShape output_shape() const noexcept { return derived().output_shape_impl(); }
  index_t total_weights() noexcept {
    if (total_weights_ < 0) {
      total_weights_ = derived().total_weights_impl();
    }
    return total_weights_;
  }

protected:
  // Total number of parameters
  index_t total_weights_{keep_dim};

  // Input & Output data. must be valid during forward and backward.
  bool has_compiled_{false};
  index_t curr_batch_size_{0};
  const_in_batch curr_x_;
  const_out_batch curr_y_;
  out_batch curr_dl_dy_;
};

template <typename Derived, typename Scalar, typename Device, typename InShape, typename OutShape, typename Blas,
          typename Par>
basic_ctx<Scalar, Device, Blas, Par, InShape, OutShape> make_ctx(
    basic_module<Derived, Scalar, Device, InShape, OutShape>& module, Blas&& b = {}, Par&& p = {}) {
  return basic_ctx<Scalar, Device, Blas, Par, InShape, OutShape>{};
}

}  // namespace mathprim::dnn
