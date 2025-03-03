#pragma once
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"
#include "mathprim/core/view.hpp"
#include "mathprim/dnn/basic_module.hpp"

namespace mathprim::dnn {

/**
 * @brief It computes Y <- X * W.T + b
 * 
 * @tparam Scalar 
 * @tparam Device 
 */
template <typename Scalar, typename Device>
class linear : public basic_module<linear<Scalar, Device>, Scalar, Device, dshape<1>, dshape<1>> {
public:
  using base = basic_module<linear<Scalar, Device>, Scalar, Device, dshape<1>, dshape<1>>;
  friend base;
  using in_batch = typename base::in_batch;
  using out_batch = typename base::out_batch;
  using const_in_batch = typename base::const_in_batch;
  using const_out_batch = typename base::const_out_batch;
  using compile_return_t = typename base::compile_return_t;
  template <typename Blas, typename ParImpl>
  using ctx_t = typename base::template ctx_t<Blas, ParImpl>;

  using in_shape = typename base::in_shape;
  using out_shape = typename base::out_shape;
  using out_buffer = to_buffer_t<out_batch>;

  using weight_matrix_buffer = contiguous_matrix_buffer<Scalar, Device>;
  using bias_buffer = contiguous_vector_buffer<Scalar, Device>;
  using weight_matrix_view = contiguous_matrix_view<Scalar, Device>;
  using bias_view = contiguous_vector_view<Scalar, Device>;

  linear() = default;
  MATHPRIM_INTERNAL_MOVE(linear, default);

  linear(index_t in_features, index_t out_features, bool has_bias = true) :
      in_features_(in_features), out_features_(out_features), has_bias_(has_bias) {
    prepare_parameters();
  }

  template <typename Blas, typename ParImpl>
  compile_return_t compile_impl(ctx_t<Blas, ParImpl>& c) {
    prepare_parameters();
    const index_t b = base::curr_batch_size_;
    y_ = make_buffer<Scalar, Device>(b, out_features_);
    dl_dy_ = make_buffer<Scalar, Device>(b, out_features_);

    auto dldw_info = c.push(W_.view().flatten(), "W");
    dL_dW_ = dldw_info.gradient().reshape(W_.shape());
    if (has_bias_) {
      auto dldb_info = c.push(b_.view(), "b");
      dL_db_ = dldb_info.gradient();
    }

    return {y_.const_view(), dl_dy_.view()};
  }

  template <typename Blas, typename ParImpl>
  void zero_grad_impl(ctx_t<Blas, ParImpl>& /* c */) {
    dl_dy_.fill_bytes(0);
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Computes
  ///////////////////////////////////////////////////////////////////////////////
  template <typename Blas, typename ParImpl>
  out_batch forward_impl(ctx_t<Blas,ParImpl>& c) {
    // x: [B, in], y: [B, out]
    const auto& x = base::curr_x_;
    auto w_t = W_.view().transpose();
    auto y = y_.view();
    auto& bl = c.blas();
    bl.gemm(1.0, x, w_t, 0.0, y);
    // TODO: add bias.
    return y;
  }

  template <typename Blas, typename ParImpl>
  void backward_impl(ctx_t<Blas, ParImpl>& c, in_batch dl_dx) {
    // dl_dw: [in, out], dl_dx: [B, in]
    // dLdW=(dLdY)T⋅XdWdL​=(dYdL​)T⋅X
    // dLdX=dLdY⋅WdXdL​=dYdL​⋅W
    auto w = W_.view();
    auto x = base::curr_x_;
    auto dl_dy = dl_dy_.const_view();  // [B, out]
    auto dl_dy_t = dl_dy.transpose();
    auto& bl = c.blas();
    bl.gemm(1.0, dl_dy_t, x, 1.0, dL_dW_);
    if (dl_dx) {
      bl.gemm(1.0, dl_dy, w, 1.0, dl_dx);
    }
    // TODO: bias.
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Meta data
  ///////////////////////////////////////////////////////////////////////////////
  in_shape input_shape_impl() const noexcept { return make_shape(in_features_); }
  out_shape out_shape_impl() const noexcept { return make_shape(out_features_); }
  index_t total_weights_impl() {
    index_t cnt = in_features_ * out_features_;
    if (has_bias_) {
      cnt += out_features_;
    }
    return cnt;
  }

  
  ///////////////////////////////////////////////////////////////////////////////
  /// Weights and biases
  ///////////////////////////////////////////////////////////////////////////////
  weight_matrix_view mat() { return W_.view(); }
  bias_view bias() { return b_.view(); }

private:
  void prepare_parameters() {
    MATHPRIM_INTERNAL_CHECK_THROW((in_features_ > 0 && out_features_ > 0), std::invalid_argument,
                                  "Invalid input/output features for linear layer.");
    if (!W_) {
      W_ = make_buffer<Scalar, Device>(out_features_, in_features_);
    }
    if (has_bias_ && !b_) {
      b_ = make_buffer<Scalar, Device>(out_features_);
    }
  }

  weight_matrix_buffer W_;  // [out, in]
  bias_buffer b_;           // [out,]
  out_buffer y_;            // [batch, out]
  out_buffer dl_dy_;        // [batch, out]

  weight_matrix_view dL_dW_;
  bias_view dL_db_;
  index_t in_features_{0}, out_features_{0};
  bool has_bias_;
};

}  // namespace mathprim::dnn
