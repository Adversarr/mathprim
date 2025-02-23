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

  MATHPRIM_INTERNAL_CHECK_THROW(am == bn, shape_error, "blas::gemm: A.shape(1) != B.shape(0)");
  MATHPRIM_INTERNAL_CHECK_THROW(an == cn, shape_error, "blas::gemm: A.shape(0) != C.shape(0)");
  MATHPRIM_INTERNAL_CHECK_THROW(bm == cm, shape_error, "blas::gemm: B.shape(1) != C.shape(1)");
}

template <index_t arow, index_t acol, index_t xrow, index_t yrow>
void check_mv_shapes(const shape_t<arow, acol> &a, const shape_t<xrow> &x, const shape_t<yrow> &y) {
  auto [an, am] = a;
  auto [xn] = x;
  auto [yn] = y;

  MATHPRIM_INTERNAL_CHECK_THROW(am == xn, shape_error, "blas::gemv: A.shape(1) != x.shape(0)");
  MATHPRIM_INTERNAL_CHECK_THROW(an == yn, shape_error, "blas::gemv: A.shape(0) != y.shape(0)");
}

template <typename Scalar, index_t... sshape_values, index_t... sstride_values>
constexpr bool is_capable_vector(const shape_t<sshape_values...> &shape,
                                 const stride_t<sstride_values...> &stride) noexcept {
  // Only the last stride can vary.
  const index_t last_stride = stride.template get<-1>();
  const index_t last_stride_elem = last_stride;
  const auto default_stride = make_default_stride<Scalar>(shape).to_array();

  return last_stride_elem * default_stride == stride.to_array();
}

template <typename Scalar, index_t srows, index_t scols, index_t lda, index_t elem>
constexpr bool is_capable_matrix(const shape_t<srows, scols> &shape, const stride_t<lda, elem> &stride) noexcept {
  auto [rows, cols] = shape;
  auto [lda_val, elem_val] = stride;
  return elem_val == 1 && lda_val >= cols;
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

template <typename Derived, typename Scalar, typename Dev>
struct basic_blas {
  using scalar_type = Scalar;
  using device_type = Dev;
  using impl_type = Derived;

  template <typename sshape, typename sstride>
  using view_type = basic_view<Scalar, sshape, sstride, Dev>;
  template <typename sshape, typename sstride>
  using const_type = basic_view<const Scalar, sshape, sstride, Dev>;

  /**
   * @brief Copy the elements of src to dst.
   *
   * @param dst
   * @param src
   */
  template <typename sshape_dst, typename sstride_dst, typename sshape_src, typename sstride_src>
  MATHPRIM_NOINLINE void copy(view_type<sshape_dst, sstride_dst> dst, const_type<sshape_src, sstride_src> src) {
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(dst), std::runtime_error,
                                  "Incompatible views for BLAS copy. (dst)");
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(src), std::runtime_error,
                                  "Incompatible views for BLAS copy. (src)");

    // assert shape equal.
    MATHPRIM_INTERNAL_CHECK_THROW(dst.shape() == src.shape(), shape_error, "blas::copy: dst.shape() != src.shape()");

    static_cast<Derived *>(this)->copy_impl(dst.flatten(), src.flatten());
  }

  /**
   * @brief Scale the elements of x by alpha.
   *
   * @param alpha
   * @param x
   */
  template <typename sshape, typename sstride>
  MATHPRIM_NOINLINE void scal(Scalar alpha, view_type<sshape, sstride> x) {
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                  "Incompatible views for BLAS scal.");

    static_cast<Derived *>(this)->scal_impl(alpha, x.flatten());
  }

  /**
   * @brief Swap the elements of x and y.
   *
   * @param x
   * @param y
   */
  template <typename sshape, typename sstride>
  MATHPRIM_NOINLINE void swap(view_type<sshape, sstride> x, view_type<sshape, sstride> y) {
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                  "Incompatible views for BLAS swap. (x)");
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                  "Incompatible views for BLAS swap. (y)");

    // assert shape equal.
    MATHPRIM_INTERNAL_CHECK_THROW(x.shape() == y.shape(), shape_error, "blas::swap: x.shape() != y.shape()");

    static_cast<Derived *>(this)->swap_impl(x.flatten(), y.flatten());
  }

  /**
   * @brief Compute y <- alpha * x + y.
   *
   * @param alpha
   * @param x
   * @param y
   */
  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  MATHPRIM_NOINLINE void axpy(Scalar alpha, const_type<sshape_x, sstride_x> x, view_type<sshape_y, sstride_y> y) {
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                  "Incompatible views for BLAS axpy. (x)");
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                  "Incompatible views for BLAS axpy. (y)");

    MATHPRIM_INTERNAL_CHECK_THROW(x.shape() == y.shape(), shape_error, "blas::axpy: x.shape() != y.shape()");

    static_cast<Derived *>(this)->axpy_impl(alpha, x.flatten(), y.flatten());
  }

  /**
   * @brief Compute the dot product of x and y.
   *
   * @param x
   * @param y
   * @return Scalar
   */
  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  MATHPRIM_NOINLINE Scalar dot(const_type<sshape_x, sstride_x> x, const_type<sshape_y, sstride_y> y) {
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                  "Incompatible views for BLAS dot. (x)");
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                  "Incompatible views for BLAS dot. (y)");

    MATHPRIM_INTERNAL_CHECK_THROW(x.shape() == y.shape(), shape_error, "blas::dot: x.shape() != y.shape()");

    return static_cast<Derived *>(this)->dot_impl(x.flatten(), y.flatten());
  }

  /**
   * @brief Compute the norm of x.
   *
   * @param x
   * @return Scalar
   */
  template <typename sshape, typename sstride>
  MATHPRIM_NOINLINE Scalar norm(const_type<sshape, sstride> x) {
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                  "Incompatible views for BLAS norm.");

    return static_cast<Derived *>(this)->norm_impl(x.flatten());
  }

  template <typename sshape, typename sstride>
  MATHPRIM_NOINLINE Scalar asum(const_type<sshape, sstride> x) {
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                  "Incompatible views for BLAS asum.");

    return static_cast<Derived *>(this)->asum_impl(x.flatten());
  }

  /**
   * @brief Compute the index of the maximum element of x.
   *
   * @param x
   * @return index_t
   */
  template <typename sshape, typename sstride>
  MATHPRIM_NOINLINE index_t amax(const_type<sshape, sstride> x) {
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                  "Incompatible views for BLAS amax.");

    return static_cast<Derived *>(this)->amax_impl(x.flatten());
  }

  /**
   * @brief Computes Y <- X * Y, element-wise.
   * @param x
   * @return index_t
   */
  template <typename SshapeX, typename SstrideX, typename SshapeY, typename SstrideY>
  MATHPRIM_NOINLINE void emul(const_type<SshapeX, SstrideX> x, view_type<SshapeY, SstrideY> y) {
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                  "Incompatible views for BLAS emul. (x)");
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                  "Incompatible views for BLAS emul. (y)");

    MATHPRIM_INTERNAL_CHECK_THROW(x.shape() == y.shape(), shape_error, "blas::emul: x.shape() != y.shape()");

    static_cast<Derived *>(this)->emul_impl(x.flatten(), y.flatten());
  }

  /**
   * @brief Computes Y <- alpha * A @ X + beta * Y
   *
   * @param alpha
   * @param mat_a if transpose, then A^T is used.
   * @param x
   * @param beta
   * @param y
   */
  template <typename sshape_A, typename sstride_A, typename sshape_x, typename sstride_x, typename sshape_y,
            typename sstride_y>
  MATHPRIM_NOINLINE void gemv(Scalar alpha, const_type<sshape_A, sstride_A> mat_a, const_type<sshape_x, sstride_x> x,
                              Scalar beta, view_type<sshape_y, sstride_y> y) {
    // check for shapes
    internal::check_mv_shapes(mat_a.shape(), x.shape(), y.shape());
    // check for capabilities
    auto mat_a_op = internal::get_matrix_op(mat_a);
    MATHPRIM_INTERNAL_CHECK_THROW(mat_a_op != internal::matrix_op::invalid, std::runtime_error,
                                  "Incompatible views for BLAS gemv. (mat_a)");
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                  "Incompatible views for BLAS gemv. (x)");
    MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                  "Incompatible views for BLAS gemv. (y)");

    // Run the actual implementation
    static_cast<Derived *>(this)->gemv_impl(alpha, mat_a, x.flatten(), beta, y.flatten());
  }

  /**
   * @brief Computes C <- alpha * A @ B + beta * C
   *
   * @param alpha
   * @param A if transpose, then A^T is used.
   * @param B if transpose, then B^T is used.
   * @param beta
   * @param C if transpose, then C^T is used.
   */
  template <typename sshape_A, typename sstride_A, typename sshape_B, typename sstride_B, typename sshape_C,
            typename sstride_C>
  MATHPRIM_NOINLINE void gemm(Scalar alpha, const_type<sshape_A, sstride_A> A, const_type<sshape_B, sstride_B> B,
                              Scalar beta, view_type<sshape_C, sstride_C> C) {
    // check for shapes
    internal::check_mm_shapes(A.shape(), B.shape(), C.shape());
    // check for capabilities
    auto a_op = internal::get_matrix_op(A);
    auto b_op = internal::get_matrix_op(B);
    auto c_op = internal::get_matrix_op(C);
    MATHPRIM_INTERNAL_CHECK_THROW(a_op != internal::matrix_op::invalid, std::runtime_error,
                                  "Incompatible views for BLAS gemm. (A)");
    MATHPRIM_INTERNAL_CHECK_THROW(b_op != internal::matrix_op::invalid, std::runtime_error,
                                  "Incompatible views for BLAS gemm. (B)");
    MATHPRIM_INTERNAL_CHECK_THROW(c_op != internal::matrix_op::invalid, std::runtime_error,
                                  "Incompatible views for BLAS gemm. (C)");

    static_cast<Derived *>(this)->gemm_impl(alpha, A, B, beta, C);
  }
};

}  // namespace mathprim::blas
