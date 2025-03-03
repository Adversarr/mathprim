#include "mathprim/blas/cpu_blas.hpp"
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
using act = dnn::activation<float, device::cpu, dshape<1>, dnn::sigmoid_activation>;
using mlp_t = dnn::sequential<linear, act, linear, act, linear, act, linear>;
using ctx_t = dnn::basic_ctx<float, device::cpu, blas::cpu_blas<float>, par::seq, dshape<1>, dshape<1>>;

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
    eigen_support::cmap(ctx_.input()).setRandom();

    ctx_.zero_grad(inr_);
    ctx_.forward(inr_);
    index_t batch_size = ctx_.input().shape(0);
    // loss = 1/2 ||Y - sin(x) sin(y)||^2
    ctx_.parallel().run(make_shape(ctx_.input().shape(0)), 
      [dl_dy = ctx_.output_gradient(),
       y = ctx_.output(),
       x = ctx_.input(), batch_size](index_t bi) {
      dl_dy(bi, 0) = (y(bi, 0) - sin(x(bi, 0)) * sin(x(bi, 1))) / batch_size;
    });
    float nrm = ctx_.blas().norm(ctx_.output_gradient());
    accumulate_loss(0.5 * nrm * nrm / batch_size);
    ctx_.backward(inr_);
    copy(fused_gradients(), ctx_.params_gradient());
  }
};


index_t W = 64;

int main () {
  srand(3407);
  mlp_t inr{
    linear(2, W),
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

  index_t batch_size = 1 << 8;
  ctx.compile(inr, batch_size); // batchsize=128
  opt o(ctx, inr);
  o.setup();

  optim::adamw_optimizer<float, device::cpu, blas::cpu_blas<float>> optimizer;
  optimizer.stopping_criteria_.max_iterations_ = 10000;
  optimizer.stopping_criteria_.tol_grad_ = 0; // never stop;
  optimizer.learning_rate_ = 1e-3;
  optimizer.beta1_ = 0.9;
  optimizer.beta2_ = 0.95;
  optimizer.weight_decay_ = 1e-2;
  auto start_time = std::chrono::high_resolution_clock::now();
  auto res = optimizer.optimize(o, [&](auto res) {
    if (res.iterations_ % 100 == 0) {
      std::cout << res << std::endl;
    }
  });

  std::cout << res << std::endl;
  auto end_time = std::chrono::high_resolution_clock::now();
  auto ms_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_time - start_time)
                     .count();

  std::cout << "Time: " << ms_time << "ms" << std::endl;
  std::cout << batch_size * res.iterations_ << " samples in " << ms_time << "ms"
            << std::endl;
  std::cout << "=> " << (batch_size * res.iterations_) / (ms_time / 1000.0)
            << " samples per second" << std::endl;

  index_t width = 40;
  ctx.compile(inr, width * width);
  auto gt = make_buffer<float>(width, width);
  ctx.parallel().run(make_shape(width, width), [in = ctx.input(), width] (auto ij) {
    auto [i, j] = ij;
    auto idx = i * width + j;
    in(idx, 0) = i * 0.5f / width - 0.5f;
    in(idx, 1) = j * 0.5f / width - 0.5f;
  });
  ctx.forward(inr);

  auto err = make_buffer<float>(width, width);
  ctx.parallel().run(make_shape(width, width), [out = ctx.output(), gt = gt.view(), err = err.view(), width] (auto ij) {
    auto [i, j] = ij;
    gt(ij) = sin(i * .5f / width - 0.5f) * sin(j * .5f / width - 0.5f);
    err(ij) = out(ij) - gt(ij);
  });

  auto rmse = ctx.blas().norm(err.view()) / width;
  auto amax = ctx.blas().amax(err.view());
  std::cout << "Max error: " << err.view().flatten()[amax] << std::endl;
  std::cout << "RMSE: " << rmse << std::endl;
  return 0;
}
