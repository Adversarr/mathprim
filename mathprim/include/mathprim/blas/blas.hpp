#pragma once
#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"

namespace mathprim::blas {

namespace internal {

template <index_t arow, index_t acol, index_t brow, index_t bcol, index_t crow, index_t ccol>
void check_mm_shapes(const shape_t<arow, acol> &a, const shape_t<brow, bcol> &b, const shape_t<crow, ccol> &c) {
  auto [an, am] = a;
  auto [bn, bm] = b;
  auto [cn, cm] = c;

  if (am != bn) {
    throw shape_error("blas::gemm: A.shape(1) != B.shape(0)");
  } else if (an != cn) {
    throw shape_error("blas::gemm: A.shape(0) != C.shape(0)");
  } else if (bm != cm) {
    throw shape_error("blas::gemm: B.shape(1) != C.shape(1)");
  }
}

template <index_t arow, index_t acol, index_t xrow, index_t yrow>
void check_mv_shapes(const shape_t<arow, acol> &a, const shape_t<xrow> &x, const shape_t<yrow> &y) {
  auto [an, am] = a;
  auto [xn] = x;
  auto [yn] = y;

  if (am != xn) {
    throw shape_error("blas::gemv: A.shape(1) != x.shape(0)");
  } else if (an != yn) {
    throw shape_error("blas::gemv: A.shape(0) != y.shape(0)");
  }
}

template <typename Scalar, index_t... sshape_values, index_t... sstride_values>
constexpr bool is_capable_vector(const shape_t<sshape_values...> &shape,
                                 const stride_t<sstride_values...> &stride) noexcept {
  // Only the last stride can vary.
  constexpr index_t scalar_size = static_cast<index_t>(sizeof(Scalar));
  const index_t last_stride = stride.template get<-1>();
  const index_t last_stride_elem = last_stride / scalar_size;
  const auto default_stride = make_default_stride<Scalar>(shape).to_array();

  return last_stride_elem * default_stride == stride.to_array();
}

template <typename Scalar, index_t srows, index_t scols, index_t lda, index_t elem>
constexpr bool is_capable_matrix(const shape_t<srows, scols> &shape, const stride_t<lda, elem> &stride) noexcept {
  constexpr index_t scalar_size = static_cast<index_t>(sizeof(Scalar));
  auto [rows, cols] = shape;
  auto [lda_val, elem_val] = stride;
  return elem_val == scalar_size && lda_val >= cols * scalar_size;
}

template <typename Scalar, typename sshape, typename sstride, typename device>
constexpr bool is_capable_vector(basic_view<Scalar, sshape, sstride, device> view) noexcept {
  return is_capable_vector<Scalar>(view.shape(), view.stride());
}

enum class matrix_op {
  invalid,
  none,       // CBLAS_TRANSPOSE::CblasNoTrans
  transpose,  // CBLAS_TRANSPOSE::CblasTrans
};

template <typename Scalar, typename sshape, typename sstride, typename device>
constexpr matrix_op get_matrix_op(basic_view<Scalar, sshape, sstride, device> view) noexcept {
  bool is_none_ok = is_capable_matrix<Scalar>(view.shape(), view.stride());
  auto transpose_view = view.transpose();
  bool is_transpose_ok = is_capable_matrix<Scalar>(transpose_view.shape(), transpose_view.stride());

  if (is_none_ok) {
    return matrix_op::none;
  } else if (is_transpose_ok) {
    return matrix_op::transpose;
  } else {
    return matrix_op::invalid;
  }
}

}  // namespace internal

template <typename Derived, typename Scalar, typename dev>
struct basic_blas {
  template <typename sshape, typename sstride>
  using view_type = basic_view<Scalar, sshape, sstride, dev>;
  template <typename sshape, typename sstride>
  using const_type = basic_view<const Scalar, sshape, sstride, dev>;

  template <typename sshape_dst, typename sstride_dst, typename sshape_src, typename sstride_src>
  void copy(view_type<sshape_dst, sstride_dst> dst, const_type<sshape_src, sstride_src> src) {
    if (!internal::is_capable_vector(dst)) {
      throw std::runtime_error("Incompatible views for BLAS copy. (dst)");
    } else if (!internal::is_capable_vector(src)) {
      throw std::runtime_error("Incompatible views for BLAS copy. (src)");
    }

    // assert shape equal.
    if (dst.shape() != src.shape()) {
      throw shape_error("blas::copy: dst.shape() != src.shape()");
    }

    static_cast<Derived *>(this)->copy_impl(dst, src);
  }

  template <typename sshape, typename sstride>
  void scal(Scalar alpha, view_type<sshape, sstride> x) {
    if (!internal::is_capable_vector(x)) {
      throw std::runtime_error("Incompatible views for BLAS scal.");
    }

    static_cast<Derived *>(this)->scal_impl(alpha, x);
  }

  template <typename sshape, typename sstride>
  void swap(view_type<sshape, sstride> x, view_type<sshape, sstride> y) {
    if (!internal::is_capable_vector(x)) {
      throw std::runtime_error("Incompatible views for BLAS swap. (x)");
    } else if (!internal::is_capable_vector(y)) {
      throw std::runtime_error("Incompatible views for BLAS swap. (y)");
    }

    // assert shape equal.
    if (x.shape() != y.shape()) {
      throw shape_error("blas::swap: x.shape() != y.shape()");
    }

    static_cast<Derived *>(this)->swap_impl(x, y);
  }

  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  void axpy(Scalar alpha, const_type<sshape_x, sstride_x> x, view_type<sshape_y, sstride_y> y) {
    if (!internal::is_capable_vector(x)) {
      throw std::runtime_error("Incompatible views for BLAS axpy. (x)");
    } else if (!internal::is_capable_vector(y)) {
      throw std::runtime_error("Incompatible views for BLAS axpy. (y)");
    }

    if (x.shape() != y.shape()) {
      throw shape_error("blas::axpy: x.shape() != y.shape()");
    }

    static_cast<Derived *>(this)->axpy_impl(alpha, x, y);
  }

  template <typename sshape_A, typename sstride_A, typename sshape_x, typename sstride_x, typename sshape_y,
            typename sstride_y>
  Scalar dot(const_type<sshape_x, sstride_x> x, const_type<sshape_y, sstride_y> y) {
    if (!internal::is_capable_vector(x)) {
      throw std::runtime_error("Incompatible views for BLAS dot. (x)");
    } else if (!internal::is_capable_vector(y)) {
      throw std::runtime_error("Incompatible views for BLAS dot. (y)");
    }

    if (x.shape() != y.shape()) {
      throw shape_error("blas::dot: x.shape() != y.shape()");
    }

    return static_cast<Derived *>(this)->dot_impl(x, y);
  }

  template <typename sshape, typename sstride>
  Scalar norm(const_type<sshape, sstride> x) {
    if (!internal::is_capable_vector<Scalar>(x)) {
      throw std::runtime_error("Incompatible views for BLAS norm.");
    }

    return static_cast<Derived *>(this)->norm_impl(x);
  }

  template <typename sshape, typename sstride>
  Scalar asum(const_type<sshape, sstride> x) {
    if (!internal::is_capable_vector(x)) {
      throw std::runtime_error("Incompatible views for BLAS asum.");
    }

    return static_cast<Derived *>(this)->asum_impl(x);
  }

  template <typename sshape, typename sstride>
  index_t amax(const_type<sshape, sstride> x) {
    if (!internal::is_capable_vector(x)) {
      throw std::runtime_error("Incompatible views for BLAS amax.");
    }

    return static_cast<Derived *>(this)->amax_impl(x);
  }

  // Y <- alpha * A * X + beta * Y
  template <typename sshape_a, typename sstride_a, typename sshape_x, typename sstride_x, typename sshape_y,
            typename sstride_y>
  void emul(Scalar alpha, const_type<sshape_a, sstride_a> a, const_type<sshape_x, sstride_x> x, Scalar beta,
            view_type<sshape_y, sstride_y> y) {
    if (!internal::is_capable_vector(a)) {
      throw std::runtime_error("Incompatible views for BLAS emul. (a)");
    } else if (!internal::is_capable_vector(x)) {
      throw std::runtime_error("Incompatible views for BLAS emul. (x)");
    } else if (!internal::is_capable_vector(y)) {
      throw std::runtime_error("Incompatible views for BLAS emul. (y)");
    }

    static_cast<Derived *>(this)->emul_impl(alpha, a, x, beta, y);
  }

  template <typename sshape_A, typename sstride_A, typename sshape_x, typename sstride_x, typename sshape_y,
            typename sstride_y>
  void gemv(Scalar alpha, const_type<sshape_A, sstride_A> mat_a, const_type<sshape_x, sstride_x> x, Scalar beta,
            view_type<sshape_y, sstride_y> y) {
    // check for shapes
    internal::check_mv_shapes(mat_a.shape(), x.shape(), y.shape());
    // check for capabilities
    auto mat_a_op = internal::get_matrix_op(mat_a);
    if (mat_a_op == internal::matrix_op::invalid) {
      throw std::runtime_error("Incompatible views for BLAS gemv. (mat_a)");
    } else if (!internal::is_capable_vector(x)) {
      throw std::runtime_error("Incompatible views for BLAS gemv. (x)");
    } else if (!internal::is_capable_vector(y)) {
      throw std::runtime_error("Incompatible views for BLAS gemv. (y)");
    };

    // Run the actual implementation
    static_cast<Derived *>(this)->gemv_impl(alpha, mat_a, x, beta, y);
  }

  template <typename sshape_A, typename sstride_A, typename sshape_B, typename sstride_B, typename sshape_C,
            typename sstride_C>
  void gemm(Scalar alpha, const_type<sshape_A, sstride_A> A, const_type<sshape_B, sstride_B> B, Scalar beta,
            view_type<sshape_C, sstride_C> C) {
    // check for shapes
    internal::check_mm_shapes(A.shape(), B.shape(), C.shape());
    // check for capabilities
    auto a_op = internal::get_matrix_op(A);
    auto b_op = internal::get_matrix_op(B);
    auto c_op = internal::get_matrix_op(C);
    if (a_op == internal::matrix_op::invalid) {
      throw std::runtime_error("Incompatible views for BLAS gemm. (A)");
    } else if (b_op == internal::matrix_op::invalid) {
      throw std::runtime_error("Incompatible views for BLAS gemm. (B)");
    } else if (c_op == internal::matrix_op::invalid) {
      throw std::runtime_error("Incompatible views for BLAS gemm. (C)");
    }

    static_cast<Derived *>(this)->gemm_impl(alpha, A, B, beta, C);
  }
};

}  // namespace mathprim::blas
