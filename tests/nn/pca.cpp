#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/dnn/nn/linear.hpp"
#include "mathprim/dnn/nn/sequential.hpp"
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/parallel/parallel.hpp"
#include <iostream>

using namespace mathprim;

using linear = dnn::linear<float, device::cpu>;
using inr_t = dnn::sequential<linear, linear>;
using ctx_t = dnn::basic_ctx<float, device::cpu, blas::cpu_eigen<float>, par::seq, dshape<1>, dshape<1>>;

Eigen::Matrix3f projection_matrix() {
  auto project_out_111 = [](const Eigen::Vector3f& v) {
    return v - v.dot(Eigen::Vector3f::Ones()) / 3 * Eigen::Vector3f::Ones();
  };

  Eigen::Matrix3f P;
  P.col(0) = project_out_111(Eigen::Vector3f::UnitX());
  P.col(1) = project_out_111(Eigen::Vector3f::UnitY());
  P.col(2) = project_out_111(Eigen::Vector3f::UnitZ());
  return P;
}

struct opt : public optim::basic_problem<opt, float, device::cpu> {
  ctx_t& ctx;
  inr_t& pca;
  opt(ctx_t& ctx, inr_t& pca) : ctx(ctx), pca(pca) {
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
    eigen_support::cmap(ctx.input()).setRandom();
    ctx.zero_grad(pca);
    ctx.forward(pca);

    // loss = 1/2 ||Y - P(X)||^2
    // dL/dY = Y - P(X)
    auto dl_dy = ctx.output_gradient();
    auto y = ctx.output();
    auto x = ctx.input();
    float loss = 0;
    ctx.parallel().run(make_shape(4), [x, y, dl_dy, &loss](index_t i) {
      auto xi = eigen_support::cmap(x[i]);
      auto yi = eigen_support::cmap(y[i]);
      auto gt = (xi - xi.dot(Eigen::Vector3f::Ones()) * Eigen::Vector3f::Ones() / 3 + Eigen::Vector3f::Random() * 2e-2)
                    .eval();
      auto dl_dyi = eigen_support::cmap(dl_dy[i]);
      dl_dyi = yi - gt;
      loss += 0.5 * dl_dyi.squaredNorm();
    });
    this->accumulate_loss(loss);
    ctx.backward(pca);

    copy(fused_gradients(), ctx.params_gradient());
  }
};

int main () {
  inr_t pca(linear(3, 2), linear(2, 3)); // 3 -> 2 -> 3
  ctx_t ctx;
  auto mat0 = eigen_support::cmap(pca.get<0>().mat()).setRandom(); // 3, 2
  auto mat1 = eigen_support::cmap(pca.get<1>().mat()).setRandom(); // 2, 3
  ctx.compile(pca, 4); // batchsize=4
  opt o(ctx, pca);
  o.setup();

  optim::adamw_optimizer<float, device::cpu, blas::cpu_eigen<float>> optimizer;
  optimizer.stopping_criteria_.max_iterations_ = 10000;
  optimizer.learning_rate_ = 1e-4;
  optimizer.beta1_ = 0.9;
  optimizer.beta2_ = 0.95;
  optimizer.weight_decay_ = 1e-3;
  std::cout << optimizer.optimize(o, [&](auto res) {
    if (res.iterations_ % 100 == 0) {
      std::cout << res << std::endl;
    }
  }) << std::endl;
  std::cout << mat0 * mat1 << std::endl;
  std::cout << projection_matrix() << std::endl;
  return 0;
}
