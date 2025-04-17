#pragma once

#include <cmath>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/defines.hpp"

namespace mathprim {
namespace blas {

template <typename T>
struct cpu_handmade : public basic_blas<cpu_handmade<T>, T, device::cpu> {
  // Level 1
  template <typename Sshape, typename Sstride>
  using view_type = basic_view<T, Sshape, Sstride, device::cpu>;
  template <typename Sshape, typename Sstride>
  using const_type = basic_view<const T, Sshape, Sstride, device::cpu>;
  using base = basic_blas<cpu_handmade<T>, T, device::cpu>;
  friend base;
  using Scalar = T;

protected:
  template <typename SshapeDst, typename SstrideDst, typename SshapeSrc, typename SstrideSrc>
  void copy_impl(const view_type<SshapeDst, SstrideDst>& dst, const const_type<SshapeSrc, SstrideSrc>& src) {
    auto shape = src.shape();
    if (src.is_contiguous() && dst.is_contiguous()) {
      std::memcpy(dst.data(), src.data(), src.size() * sizeof(T));
    } else {
      for (auto sub : shape) {
        dst(sub) = src(sub);
      }
    }
  }

  template <typename Sshape, typename Sstride>
  void scal_impl(const T& alpha, const view_type<Sshape, Sstride>& src) {
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

  template <typename SshapeSrc, typename SstrideSrc, typename SshapeDst, typename SstrideDst>
  void swap_impl(const view_type<SshapeSrc, SstrideSrc>& src, const view_type<SshapeDst, SstrideDst>& dst) {
    auto shape = src.shape();
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      ::std::swap(src(sub), dst(sub));
    }
  }

  template <typename SshapeX, typename SstrideX, typename SshapeY, typename SstrideY>
  void axpy_impl(T alpha, const const_type<SshapeX, SstrideX>& x, view_type<SshapeY, SstrideY> y) {
    auto shape = x.shape();
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      y(sub) += alpha * x(sub);
    }
  }

  template <typename SshapeX, typename SstrideX, typename SshapeY, typename SstrideY>
  T dot_impl(const const_type<SshapeX, SstrideX>& x, const const_type<SshapeY, SstrideY>& y) {
    auto shape = x.shape();
    T result = 0;
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      result += x(sub) * y(sub);
    }
    return result;
  }

  template <typename Sshape, typename Sstride>
  T norm_impl(const const_type<Sshape, Sstride>& x) {
    auto shape = x.shape();
    T result = 0;
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      result += x(sub) * x(sub);
    }
    return std::sqrt(result);
  }

  template <typename Sshape, typename Sstride>
  T asum_impl(const const_type<Sshape, Sstride>& x) {
    auto shape = x.shape();
    T result = 0;
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      result += std::abs(x(sub));
    }
    return result;
  }

  template <typename Sshape, typename Sstride>
  index_t amax_impl(const const_type<Sstride, Sshape>& x) {
    auto shape = x.shape();
    Scalar maximum = std::abs(x[0]);
    index_t index = 0;
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto sub : shape) {
      auto ax = std::abs(x(sub));
      if (ax > maximum) {
        maximum = ax;
        index = sub[0];
      }
    }
    return index;
  }

  // Y <- alpha * A * X + beta * Y
  template <typename SshapeX, typename SstrideX, typename SshapeY, typename SstrideY>
  void inplace_emul_impl(const_type<SshapeX, SstrideX> x, view_type<SshapeY, SstrideY> y) {
    const auto total = x.shape(0);
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (index_t i = 0; i < total; ++i) {
      y[i] = x[i] * y[i];
    }
  }

  template <typename SshapeX, typename SstrideX, typename SshapeY, typename SstrideY,  //
            typename SshapeZ, typename SstrideZ>
  void emul_impl(const_type<SshapeX, SstrideX> x, const_type<SshapeY, SstrideY> y,  //
                 view_type<SshapeZ, SstrideZ> z) {
    const auto total = x.shape(0);
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (index_t i = 0; i < total; ++i) {
      z[i] = x[i] * y[i];
    }
  }

  // // Level 2
  // // y <- alpha * A * x + beta * y
  template <typename SshapeA, typename SstrideA, typename SshapeX, typename SstrideX, typename SshapeY,
            typename SstrideY>
  void gemv_impl(const T& alpha, const const_type<SshapeA, SstrideA>& A, const const_type<SshapeX, SstrideX>& x,
                 const T& beta, const view_type<SshapeY, SstrideY>& y) {
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
  template <typename SshapeA, typename SstrideA, typename SshapeB, typename SstrideB, typename SshapeC,
            typename SstrideC>
  void gemm_impl(const T& alpha, const const_type<SshapeA, SstrideA>& A, const const_type<SshapeB, SstrideB>& B,
                 const T& beta, const view_type<SshapeC, SstrideC>& C) {
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

  template <typename SshapeA, typename SstrideA, typename SshapeB, typename SstrideB, typename SshapeC,
            typename SstrideC>
  void gemm_batch_strided_impl(Scalar alpha, const_type<SshapeA, SstrideA> A,
                                           const_type<SshapeB, SstrideB> B, Scalar beta,
                                           view_type<SshapeC, SstrideC> C) {
    for (index_t i = 0; i < A.shape(0); ++i) {
      gemm_impl(alpha, A[i], B[i], beta, C[i]);
    }
  }



  template <typename SshapeX, typename SstrideX,
            typename SshapeY, typename SstrideY>
  void axpby_impl(Scalar alpha, const_type<SshapeX, SstrideX> x, Scalar beta, view_type<SshapeY, SstrideY> y) {
    auto total = x.numel();
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (index_t i = 0; i < total; ++i) {
      y[i] = alpha * x[i] + beta * y[i];
    }
  }
};

}  // namespace blas

}  // namespace mathprim
