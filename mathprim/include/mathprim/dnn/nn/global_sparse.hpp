#pragma once
#include "mathprim/dnn/basic_module.hpp"
namespace mathprim::dnn {

/**
 * @brief Y <- W * X, where W is a sparse matrix, X, Y is dense.
 *        Sparse  W: [M, N]
 *        Input   X: [Batch, N, InFeatures]
 *        Output  Y: [Batch, M, InFeatures]
 * @note this module does not provide optimizable parameters.
 * @tparam Scalar
 * @tparam Device
 * @tparam SpBlas
 */
template <typename SpBlas>
class global_sparse : public basic_module<global_sparse<SpBlas>, typename SpBlas::scalar_type,
                                          typename SpBlas::device_type, dshape<2>, dshape<2>> {
public:
  using base = basic_module<global_sparse<SpBlas>, typename SpBlas::scalar_type, typename SpBlas::device_type,
                            dshape<2>, dshape<2>>;
  friend base;
  using in_batch = typename base::in_batch;
  using out_batch = typename base::out_batch;
  using const_in_batch = typename base::const_in_batch;
  using const_out_batch = typename base::const_out_batch;
  using compile_return_t = typename base::compile_return_t;
  using scalar_t = typename base::scalar_t;
  using device_t = typename base::device_t;
  template <typename Blas, typename ParImpl>
  using ctx_t = typename base::template ctx_t<Blas, ParImpl>;

  using in_shape = typename base::in_shape;
  using out_shape = typename base::out_shape;
  using out_buffer = to_buffer_t<out_batch>;

  global_sparse(global_sparse&&) = default;

  explicit global_sparse(SpBlas op, index_t n_features, bool trans_op) :
      op_(std::move(op)), n_features_(n_features), trans_op_(trans_op) {}

  template <typename Blas, typename ParImpl>
  compile_return_t compile_impl(ctx_t<Blas, ParImpl>& /* c */) {
    auto input = base::input();
    auto batch = input.shape(0);
    auto n_points = input.shape(1);
    auto in_features = input.shape(2);
    y_ = make_buffer<scalar_t, device_t>(batch, n_points, n_features_);
    dl_dy_ = make_buffer<scalar_t, device_t>(batch, n_points, n_features_);
    return {y_.const_view(), dl_dy_.view()};
  }

  template <typename Blas, typename ParImpl>
  void zero_grad_impl(ctx_t<Blas, ParImpl>& /* c */) {
    dl_dy_.fill_bytes(0);
  }

  void reset_parameters_impl() {}


  ///////////////////////////////////////////////////////////////////////////////
  /// Computes
  ///////////////////////////////////////////////////////////////////////////////
  template <typename Blas, typename ParImpl>
  out_batch forward_impl(ctx_t<Blas, ParImpl>& /* c */) {
    // x: [B, N, in], y: [B, N, out]
    auto x = base::input();
    auto y = y_.view();
    // Use a stupid way to compute the output
    const index_t batch_size = x.shape(0);
    for (index_t i = 0; i < batch_size; ++i) {
      op_.spmm(1.0, x[i], 0.0, y[i], trans_op_);
    }
    return y;
  }

  template <typename Blas, typename ParImpl>
  void backward_impl(ctx_t<Blas, ParImpl>& /* c */, bool compute_dldx) {
    if (!compute_dldx) {
      return;
    }

    // x: [B, N, in], y: [B, N, out]
    auto x = base::input();
    auto dl_dy = dl_dy_.const_view();
    auto dl_dx = base::curr_dl_dx_;
    for (index_t i = 0; i < x.shape(0); ++i) {
      op_.spmm(1.0, dl_dy[i], 1.0, dl_dx[i], !trans_op_);
    }
  }

  in_shape input_shape_impl() const noexcept {
    auto n_points = op_.matrix().cols();
    return make_shape(n_points, n_features_);
  }

  out_shape out_shape_impl() const noexcept {
    auto n_points = op_.matrix().rows();
    return make_shape(n_points, n_features_);
  }
  index_t total_weights_impl() { return 0; }

  SpBlas& sparse() noexcept { return op_; }
  bool is_transpose() const noexcept { return trans_op_; }

private:
  SpBlas op_;
  index_t n_features_;
  out_buffer y_;
  out_buffer dl_dy_;

  bool trans_op_;
};
}  // namespace mathprim::dnn