#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/dnn/nn/activation.hpp"
#include "mathprim/dnn/nn/linear.hpp"
#include "mathprim/dnn/nn/sequential.hpp"
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/parallel/parallel.hpp"
#include <iostream>

using namespace mathprim;

using linear = dnn::linear<float, device::cpu>;
// using act = dnn::activation<float, device::cpu, dshape<1>, dnn::relu_activation>;
using act = dnn::activation<float, device::cpu, dshape<1>, dnn::sigmoid_activation>;
using mlp_t = dnn::sequential<linear, act, linear, act, linear, act, linear>;
using ctx_t = dnn::basic_ctx<float, device::cpu, blas::cpu_eigen<float>, par::seq, dshape<1>, dshape<1>>;

struct opt : public optim::basic_problem<opt, float, device::cpu> {
  ctx_t& ctx_;
  mlp_t& inr_;
  opt(ctx_t& ctx, mlp_t& pca) : ctx_(ctx), inr_(pca) {
    ctx.for_each_parameter([this](auto& param) {
      this->register_parameter(param.value(), param.name());
    });
  }

  void eval_gradients_impl() {
    // sample a batch, forward and backward
    eval_value_and_gradients_impl();
  }

  void eval_value_impl() {
    // sample a batch, forward
    eval_value_and_gradients_impl();
  }

  void eval_value_and_gradients_impl() {
    // sample_random_points(ctx.input());
    auto x = eigen_support::cmap(ctx_.input().flatten()).setRandom();
    ctx_.zero_grad(inr_);
    ctx_.forward(inr_);

    // loss = 1/2 ||Y - sin(X)||^2
    // dL/dY = Y - sin(X)
    auto dl_dy = eigen_support::cmap(ctx_.output_gradient().flatten());
    auto y = eigen_support::cmap(ctx_.output().flatten());
    auto gt = x.array().sin().matrix().eval();  // gt=sin(x)
    dl_dy = y - gt;

    
    this->accumulate_loss(0.5 * dl_dy.squaredNorm());
    ctx_.backward(inr_);
    copy(fused_gradients(), ctx_.params_gradient());
  }
};

index_t W = 16;

int main () {
  mlp_t inr{
    linear(1, W),
    act(make_shape(W)),
    linear(W, W),
    act(make_shape(W)),
    linear(W, W),
    act(make_shape(W)),
    linear(W, 1)};
  ctx_t ctx;
  eigen_support::cmap(inr.get<0>().mat()).setRandom();
  eigen_support::cmap(inr.get<2>().mat()).setRandom();
  eigen_support::cmap(inr.get<4>().mat()).setRandom();
  eigen_support::cmap(inr.get<6>().mat()).setRandom();

  ctx.compile(inr, 4); // batchsize=4
  opt o(ctx, inr);
  o.setup();

  optim::adamw_optimizer<float, device::cpu, blas::cpu_eigen<float>> optimizer;
  optimizer.stopping_criteria_.max_iterations_ = 10000;
  optimizer.learning_rate_ = 3e-4;
  optimizer.beta1_ = 0.9;
  optimizer.beta2_ = 0.99;
  std::cout << optimizer.optimize(o, [&](auto res) {
    if (res.iterations_ % 100 == 0) {
      std::cout << res << std::endl;
    }
  }) << std::endl;

  // compute the error.
  auto predict_at = [&ctx, &inr] (float x) {
    ctx.input()(0, 0) = x;
    ctx.forward(inr);
    return ctx.output()(0, 0);
  };

  float error = 0, max_err = 0;
  auto N = 100;
  for (int i = 0; i < N; ++i) {
    float x = (static_cast<float>(i) / N) - 0.5;
    float y = predict_at(x);
    auto eri = std::abs(y - std::sin(x));
    error += eri;
    max_err = std::max(max_err, eri);
  }

  std::cout << "Avg Error: " << error / N << std::endl;
  std::cout << "Max Error: " << max_err << std::endl;
  return 0;
}
