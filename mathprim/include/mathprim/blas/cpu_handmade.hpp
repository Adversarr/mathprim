#pragma once

#include <cmath>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/defines.hpp"

namespace mathprim {
namespace blas {

template <typename T>
struct cpu_handmade : public basic_blas<cpu_handmade<T>, T, device::cpu> {
  // Level 1
  template <typename sshape, typename sstride>
  using view_type = basic_view<T, sshape, sstride, device::cpu>;
  template <typename sshape, typename sstride>
  using const_type = basic_view<const T, sshape, sstride, device::cpu>;
  using base = basic_blas<cpu_handmade<T>, T, device::cpu>;
  friend base;
  using Scalar = T;

protected:
  template <typename sshape_dst, typename sstride_dst, typename sshape_src, typename sstride_src>
  void copy_impl(const view_type<sshape_dst, sstride_dst>& dst, const const_type<sshape_src, sstride_src>& src) {
    auto shape = src.shape();
    if (src.is_contiguous() && dst.is_contiguous()) {
      std::memcpy(dst.data(), src.data(), src.size() * sizeof(T));
    } else {
      for (auto sub : shape) {
        dst(sub) = src(sub);
      }
    }
  }

  template <typename sshape, typename sstride>
  void scal_impl(const T& alpha, const view_type<sshape, sstride>& src) {
    auto shape = src.shape();

    if (src.is_contiguous()) {
      auto* data = src.data();
      auto total = src.size();
      // Unroll mannually 4 times to enable vectorization
      for (index_t i = 0; i < total; i += 4) {
        data[i] = alpha * data[i];
        data[i + 1] = alpha * data[i + 1];
        data[i + 2] = alpha * data[i + 2];
        data[i + 3] = alpha * data[i + 3];
      }

      // Handle the remaining
      for (index_t i = total - total % 4; i < total; ++i) {
        data[i] = alpha * data[i];
      }
      return;
    } else {
      MATHPRIM_PRAGMA_UNROLL_HOST
      for (auto sub : shape) {
        src(sub) = alpha * src(sub);
      }
    }
  }

  template <typename sshape_src, typename sstride_src, typename sshape_dst, typename sstride_dst>
  void swap_impl(const view_type<sshape_src, sstride_src>& src, const view_type<sshape_dst, sstride_dst>& dst) {
    auto shape = src.shape();
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      ::std::swap(src(sub), dst(sub));
    }
  }

  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  void axpy_impl(T alpha, const const_type<sshape_x, sstride_x>& x, view_type<sshape_y, sstride_y> y) {
    auto shape = x.shape();
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      y(sub) += alpha * x(sub);
    }
  }

  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  T dot_impl(const const_type<sshape_x, sstride_x>& x, const const_type<sshape_y, sstride_y>& y) {
    auto shape = x.shape();
    T result = 0;
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      result += x(sub) * y(sub);
    }
    return result;
  }

  template <typename sshape, typename sstride>
  T norm_impl(const const_type<sshape, sstride>& x) {
    auto shape = x.shape();
    T result = 0;
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      result += x(sub) * x(sub);
    }
    return std::sqrt(result);
  }

  template <typename sshape, typename sstride>
  T asum_impl(const const_type<sshape, sstride>& x) {
    auto shape = x.shape();
    T result = 0;
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      result += std::abs(x(sub));
    }
    return result;
  }

  template <typename sshape, typename sstride>
  index_t amax_impl(const const_type<sstride, sshape>& x) {
    auto shape = x.shape();
    float maximum = x(*shape.begin());
    index_t index = 0;
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      if (x(sub) > maximum) {
        maximum = x(sub);
        index = sub[0];
      }
    }
    return index;
  }

  // Y <- alpha * A * X + beta * Y
  template <typename SshapeX, typename SstrideX, typename SshapeY, typename SstrideY>
  MATHPRIM_NOINLINE void emul_impl(const_type<SshapeX, SstrideX> x, view_type<SshapeY, SstrideY> y) {
    auto total = x.shape(0);
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (index_t i = 0; i < total; ++i) {
      y[i] = x[i] * y[i];
    }
  }

  // // Level 2
  // // y <- alpha * A * x + beta * y
  template <typename sshape_A, typename sstride_A, typename sshape_x, typename sstride_x, typename sshape_y,
            typename sstride_y>
  void gemv_impl(const T& alpha, const const_type<sshape_A, sstride_A>& A, const const_type<sshape_x, sstride_x>& x,
                 const T& beta, const view_type<sshape_y, sstride_y>& y) {
    auto [n, m] = A.shape();
    for (index_t i = 0; i < n; ++i) {
      T sum = 0;
      MATHPRIM_PRAGMA_UNROLL_HOST
      for (index_t j = 0; j < m; ++j) {
        sum += A(i, j) * x(j);
      }
      y(i) = alpha * sum + beta * y(i);
    }
  }

  //
  // // Level 3
  // // C <- alpha * A * B + beta * C
  template <typename sshape_A, typename sstride_A, typename sshape_B, typename sstride_B, typename sshape_C,
            typename sstride_C>
  void gemm_impl(const T& alpha, const const_type<sshape_A, sstride_A>& A, const const_type<sshape_B, sstride_B>& B,
                 const T& beta, const view_type<sshape_C, sstride_C>& C) {
    auto [m, k] = A.shape();
    auto [k2, n] = B.shape();

    // Initialize C = beta * C
    for (index_t i = 0; i < m; ++i) {
      MATHPRIM_PRAGMA_UNROLL_HOST
      for (index_t j = 0; j < n; ++j) {
        C(i, j) *= beta;
      }
    }

    // Blocking parameters (adjusted based on L1 cache size)
    constexpr index_t block_size = 16;
    T block[block_size][block_size];  // Stack memory allocation

    // Blocked computation
    for (index_t i_outer = 0; i_outer < m; i_outer += block_size) {
      const index_t i_bound = std::min(i_outer + block_size, m);

      for (index_t j_outer = 0; j_outer < n; j_outer += block_size) {
        const index_t j_bound = std::min(j_outer + block_size, n);

        // Initialize temporary block
        for (index_t i = i_outer; i < i_bound; ++i) {
          MATHPRIM_PRAGMA_UNROLL_HOST
          for (index_t j = j_outer; j < j_bound; ++j) {
            block[i - i_outer][j - j_outer] = 0;
          }
        }

        // Accumulate results into the temporary block
        for (index_t l_outer = 0; l_outer < k; l_outer += block_size) {
          const index_t l_bound = std::min(l_outer + block_size, k);

          // Core computation part
          for (index_t i = i_outer; i < i_bound; ++i) {
            for (index_t l = l_outer; l < l_bound; ++l) {
              const T a_val = A(i, l);
              MATHPRIM_PRAGMA_UNROLL_HOST
              for (index_t j = j_outer; j < j_bound; ++j) {
                block[i - i_outer][j - j_outer] += a_val * B(l, j);
              }
            }
          }
        }

        // Write results back to matrix C
        for (index_t i = i_outer; i < i_bound; ++i) {
          MATHPRIM_PRAGMA_UNROLL_HOST
          for (index_t j = j_outer; j < j_bound; ++j) {
            C(i, j) += alpha * block[i - i_outer][j - j_outer];
          }
        }
      }
    }
  }
};

}  // namespace blas

}  // namespace mathprim
