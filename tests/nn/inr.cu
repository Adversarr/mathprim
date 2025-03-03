#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/blas/cublas.cuh"
#include "mathprim/dnn/nn/activation.hpp"
#include "mathprim/dnn/nn/linear.hpp"
#include "mathprim/dnn/nn/sequential.hpp"
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/parallel/parallel.hpp"
#include <iostream>

using namespace mathprim;

using linear = dnn::linear<float, device::cuda>;
using act = dnn::activation<float, device::cuda, dshape<1>, dnn::sigmoid_activation>;
using mlp_t = dnn::sequential<linear, act, linear, act, linear, act, linear>;
using ctx_t = dnn::basic_ctx<float, device::cuda, blas::cublas<float>, par::cuda, dshape<1>, dshape<1>>;

template <typename Ctx, typename Mat> void init(Ctx &ctx, Mat m) {
  ctx.parallel().run(m.shape(), [mat = m, tp = time(nullptr)] __device__ (auto ij) {
    auto [i, j] = ij;
    auto idx = i * mat.shape(1) + j;
    // mat(i, j)
    curandState_t state;
    curand_init(tp, idx, 0, &state);
    mat(i, j) = curand_uniform(&state) * 2 - 1;
  });
}
struct opt : public optim::basic_problem<opt, float, device::cuda> {
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
    init(ctx_, ctx_.input());

    ctx_.zero_grad(inr_);
    ctx_.forward(inr_);
    // loss = 1/2 ||Y - sin(x) sin(y)||^2
    ctx_.parallel().run(make_shape(ctx_.input().shape(0)), 
      [dl_dy = ctx_.output_gradient(),
       y = ctx_.output(),
       x = ctx_.input()] __device__ (index_t bi) {
      dl_dy(bi, 0) = y(bi, 0) - sin(x(bi, 0)) * sin(x(bi, 1));
    });

    accumulate_loss(0.5 * ctx_.blas().norm(ctx_.output_gradient()));
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
  init(ctx, inr.get<0>().mat());
  init(ctx, inr.get<2>().mat());
  init(ctx, inr.get<4>().mat());
  init(ctx, inr.get<6>().mat());

  index_t batch_size = 1 << 14;
  ctx.compile(inr, batch_size); // batchsize=128
  opt o(ctx, inr);
  o.setup();

  optim::adamw_optimizer<float, device::cuda, blas::cublas<float>> optimizer;
  optimizer.stopping_criteria_.max_iterations_ = 10000;
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

  return 0;
}
