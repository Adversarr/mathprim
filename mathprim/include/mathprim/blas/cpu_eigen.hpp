#pragma once

#include <cmath>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/supports/eigen_dense.hpp"

namespace mathprim {
namespace blas {

template <typename T>
struct cpu_eigen : public basic_blas<cpu_eigen<T>, T, device::cpu> {
  // Level 1
  template <typename sshape, typename sstride>
  using view_type = basic_view<T, sshape, sstride, device::cpu>;
  template <typename sshape, typename sstride>
  using const_type = basic_view<const T, sshape, sstride, device::cpu>;

  using base = basic_blas<cpu_eigen<T>, T, device::cpu>;
  friend base;
  using Scalar = T;

protected:
  template <typename sshape_dst, typename sstride_dst, typename sshape_src, typename sstride_src>
  void copy_impl(view_type<sshape_dst, sstride_dst> dst, const_type<sshape_src, sstride_src> src) {
    auto map_dst = eigen_support::amap(dst);
    auto map_src = eigen_support::amap(src);
    map_dst.noalias() = map_src;
  }

  template <typename sshape, typename sstride>
  void scal_impl(T alpha, view_type<sshape, sstride> src) {
    auto map_src = eigen_support::amap(src);
    map_src *= alpha;
  }

  template <typename sshape_src, typename sstride_src, typename sshape_dst, typename sstride_dst>
  void swap_impl(view_type<sshape_src, sstride_src> src, view_type<sshape_dst, sstride_dst> dst) {
    auto map_src = eigen_support::amap(src);
    auto map_dst = eigen_support::amap(dst);
    if constexpr (sshape_src::ndim == 2) {
      for (index_t i = 0; i < src.shape(0); ++i) {
        for (index_t j = 0; j < src.shape(1); ++j) {
          std::swap(map_src(i, j), map_dst(i, j));
        }
      }
    } else {
      for (index_t i = 0; i < src.shape(0); ++i) {
        std::swap(map_src(i), map_dst(i));
      }
    }
  }

  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  void axpy_impl(T alpha, const_type<sshape_x, sstride_x> x, view_type<sshape_y, sstride_y> y) {
    auto map_x = eigen_support::amap(x);
    auto map_y = eigen_support::amap(y);
    map_y.noalias() += alpha * map_x;
  }

  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  T dot_impl(const_type<sshape_x, sstride_x> x, const_type<sshape_y, sstride_y> y) {
    auto map_x = eigen_support::amap(x);
    auto map_y = eigen_support::amap(y);
    return map_x.dot(map_y);
  }

  template <typename sshape, typename sstride>
  T norm_impl(const_type<sshape, sstride> x) {
    auto map_x = eigen_support::amap(x);
    return map_x.norm();
  }

  template <typename sshape, typename sstride>
  T asum_impl(const_type<sshape, sstride> x) {
    auto map_x = eigen_support::amap(x);
    return map_x.cwiseAbs().sum();
  }

  template <typename sshape, typename sstride>
  index_t amax_impl(const_type<sstride, sshape> x) {
    auto map_x = eigen_support::amap(x);
    Eigen::Index max_index;
    map_x.cwiseAbs().maxCoeff(max_index);
    return static_cast<index_t>(max_index);
  }

  // element-wise operatons
  // Y <- alpha * A * X + beta * Y
  template <typename SshapeX, typename SstrideX, typename SshapeY, typename SstrideY>
  MATHPRIM_NOINLINE void emul_impl(const_type<SshapeX, SstrideX> x, view_type<SshapeY, SstrideY> y) {
    auto map_x = eigen_support::amap(x);
    auto map_y = eigen_support::amap(y);
    map_y = map_x.cwiseProduct(map_y);
  }

  // // Level 2
  // // y <- alpha * A * x + beta * y
  template <typename sshape_A, typename sstride_A, typename sshape_x, typename sstride_x, typename sshape_y,
            typename sstride_y>
  void gemv_impl(T alpha, const_type<sshape_A, sstride_A> A, const_type<sshape_x, sstride_x> x, T beta,
                 view_type<sshape_y, sstride_y> y) {
    if (A.is_contiguous() && x.is_contiguous() && y.is_contiguous()) {
      auto map_a = eigen_support::cmap(A);
      auto map_x = eigen_support::cmap(x);
      auto map_y = eigen_support::cmap(y);
      map_y *= beta;
      map_y.noalias() += alpha * map_a.transpose() * map_x;
    } else {
      auto a_transpose = A.transpose();
      if (a_transpose.is_contiguous() && x.is_contiguous() && y.is_contiguous()) {
        auto map_a = eigen_support::cmap(a_transpose);
        auto map_x = eigen_support::cmap(x);
        auto map_y = eigen_support::cmap(y);
        map_y *= beta;
        map_y.noalias() += alpha * map_a * map_x;
      } else {
        auto map_a = eigen_support::amap(A);
        auto map_x = eigen_support::amap(x);
        auto map_y = eigen_support::amap(y);
        map_y *= beta;
        map_y.noalias() += alpha * map_a.transpose() * map_x;
      }
    }
  }

  // // Level 3
  // // C <- alpha * A * B + beta * C
  template <typename sshape_A, typename sstride_A, typename sshape_B, typename sstride_B, typename sshape_C,
            typename sstride_C>
  void gemm_impl(T alpha, const_type<sshape_A, sstride_A> A, const_type<sshape_B, sstride_B> B, T beta,
                 view_type<sshape_C, sstride_C> C) {
    if (A.is_contiguous() && B.is_contiguous() && C.is_contiguous()) {
      auto map_A = eigen_support::cmap(A);
      auto map_B = eigen_support::cmap(B);
      auto map_C = eigen_support::cmap(C);
      map_C *= beta;
      map_C.noalias() += alpha * map_B * map_A;
    } else {
      auto a_transpose = A.transpose();
      auto b_transpose = B.transpose();
      auto c_transpose = C.transpose();
      if (a_transpose.is_contiguous() && b_transpose.is_contiguous() && c_transpose.is_contiguous()) {
        auto map_A = eigen_support::cmap(a_transpose);
        auto map_B = eigen_support::cmap(b_transpose);
        auto map_C = eigen_support::cmap(c_transpose);
        map_C *= beta;
        map_C.noalias() += alpha * map_A * map_B;  // C.T += alpha * A.T * B.T <=> C += alpha * B * A
      } else {
        auto map_A = eigen_support::amap(A);
        auto map_B = eigen_support::amap(B);
        auto map_C = eigen_support::amap(C);
        map_C *= beta;
        map_C.noalias() += alpha * map_B * map_A;
      }
    }
  }
};

}  // namespace blas

}  // namespace mathprim
