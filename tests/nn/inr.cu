#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <fstream>
#include <curand_kernel.h>
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/blas/cublas.cuh"
#include "mathprim/dnn/nn/activation.hpp"
#include "mathprim/dnn/nn/linear.hpp"
#include "mathprim/dnn/nn/sequential.hpp"
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/parallel/parallel.hpp"
#include "mathprim/supports/io/npy.hpp"
#include <iostream>

using namespace mathprim;

using linear = dnn::linear<float, device::cuda>;
using act = dnn::activation<float, device::cuda, dshape<1>, dnn::relu_activation>;
using mlp_t = dnn::sequential<linear, act, linear, act, linear, act, linear>;
using ctx_t = dnn::basic_ctx<float, device::cuda, blas::cublas<float>, par::cuda, dshape<1>, dshape<1>>;

template <typename Ctx, typename Mat> void init(Ctx &ctx, Mat m) {
  auto mat_host = make_buffer<float>(m.shape());
  eigen_support::cmap(mat_host.view()).setRandom();
  copy(m, mat_host.view());
  auto [fan_out, fan_in] = m.shape();
  float scale = sqrt(6.f / (fan_in + fan_out));
  ctx.blas().scal(scale, m);
}

void init_x(ctx_t &ctx, contiguous_matrix_view<float, device::cuda> x) {
  auto bs = x.shape(0);
  index_t width = static_cast<index_t>(sqrt(bs));
  if (width * width != bs) {
    throw std::runtime_error("Batch size must be a square number.");
  }

  ctx.parallel().run(make_shape(width, width), [x, width] __device__(auto ij) {
    auto [i, j] = ij;
    x(i * width + j, 0) = (i * 2.f / width - 1.f) * M_PI;
    x(i * width + j, 1) = (j * 2.f / width - 1.f) * M_PI;
  });
}

struct opt : public optim::basic_problem<opt, float, device::cuda> {
  ctx_t& ctx_;
  mlp_t& inr_;
  opt(ctx_t& ctx, mlp_t& pca) : ctx_(ctx), inr_(pca) {
    ctx.for_each_parameter([this](auto& param) {
      this->register_parameter(param.value(), param.name());
    });

    init_x(ctx_, ctx_.input());
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
    ctx_.zero_grad(inr_);
    ctx_.forward(inr_);
    // loss = 1/2 ||Y - sin(x) sin(y)||^2
    ctx_.parallel().run(make_shape(ctx_.input().shape(0)), 
      [dl_dy = ctx_.output_gradient(),
       y = ctx_.output(),
       x = ctx_.input()] __device__ (index_t bi) {
      float batch_size = y.shape(0);
      dl_dy(bi, 0) = (y(bi, 0) - sin(x(bi, 0)) * sin(x(bi, 1))) / batch_size;
    });

    float rmse = ctx_.blas().norm(ctx_.output_gradient());
    accumulate_loss(rmse);
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
  index_t batch_size = 1 << 10;
  ctx.compile(inr, batch_size); // batchsize=128

  init(ctx, inr.get<0>().mat());
  init(ctx, inr.get<2>().mat());
  init(ctx, inr.get<4>().mat());
  init(ctx, inr.get<6>().mat());

  zeros(inr.get<0>().bias());
  zeros(inr.get<2>().bias());
  zeros(inr.get<4>().bias());
  zeros(inr.get<6>().bias());
  ctx.for_each_parameter([&ctx](auto &param){
    auto norm_val = ctx.blas().norm(param.value());
    std::cout << param.name() << " norm: " << norm_val << std::endl;
  });

  opt o(ctx, inr);
  o.setup();

  optim::adamw_optimizer<float, device::cuda, blas::cublas<float>> optimizer;
  optimizer.stopping_criteria_.max_iterations_ = 10000;
  optimizer.stopping_criteria_.tol_grad_ = 0; // never stop;
  optimizer.learning_rate_ = 1e-4;
  optimizer.beta1_ = 0.9;
  optimizer.beta2_ = 0.99;
  optimizer.weight_decay_ = 1e-4;
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

  // Testings.
  index_t width = 1 << 5;
  ctx.compile(inr, width * width);
  auto gt = make_cuda_buffer<float>(width, width);
  ctx.parallel().run(make_shape(width, width), [in = ctx.input(), gt = gt.view(), width]__device__(auto ij) {
    auto [i, j] = ij;
    auto idx = i * width + j;
    in(idx, 0) = (i * 2.f / width - 1.f) * M_PI;
    in(idx, 1) = (j * 2.f / width - 1.f) * M_PI;

    gt(i, j) = sin(in(idx, 0)) * sin(in(idx, 1));
  });
  ctx.forward(inr);

  auto err = make_cuda_buffer<float>(width, width);
  ctx.blas().copy(err.view(), ctx.output().reshape(width, width));
  ctx.blas().axpy(-1, gt.view(), err.view());

  auto rmse = ctx.blas().norm(err.view()) / width;
  std::cout << "RMSE: " << rmse << std::endl;

  // copy back to host
  auto out = make_buffer<float>(width, width);
  copy(out.view(), ctx.output().reshape(out.shape()));
  auto file = std::ofstream("sinsin.npy", std::ios::binary);
  io::numpy<float, 2>{}.write(file, out.const_view());

  ctx.for_each_parameter([&ctx](auto &param){
    auto norm_val = ctx.blas().norm(param.value());
    std::cout << param.name() << " norm: " << norm_val << std::endl;
  });
  return 0;
}
