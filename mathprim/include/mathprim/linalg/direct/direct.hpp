#pragma once
#include "mathprim/core/view.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::sparse::direct {

template <typename Derived, typename Scalar, sparse_format Format, typename Device>
class basic_direct_solver {
public:
  using vector_view = contiguous_view<Scalar, dshape<1>, Device>;
  using const_vector = contiguous_view<const Scalar, dshape<1>, Device>;
  using matrix_view = contiguous_view<Scalar, dshape<2>, Device>;
  using const_matrix_view = contiguous_view<const Scalar, dshape<2>, Device>;

  using sparse_view = basic_sparse_view<Scalar, Device, Format>;
  using const_sparse = basic_sparse_view<const Scalar, Device, Format>;

  basic_direct_solver() = default;
  basic_direct_solver(const basic_direct_solver&) = delete;
  basic_direct_solver(basic_direct_solver&&) = default;

  /**
   * @brief Solve the linear system. A x = y.
   * 
   * @param lhs x
   * @param rhs y
   */
  MATHPRIM_NOINLINE void solve(vector_view lhs, const_vector rhs) {
    MATHPRIM_INTERNAL_CHECK_THROW(mat_.cols() == lhs.size(), std::runtime_error, "Invalid size of lhs");
    MATHPRIM_INTERNAL_CHECK_THROW(mat_.rows() == rhs.size(), std::runtime_error, "Invalid size of rhs");
    static_cast<Derived*>(this)->solve_impl(lhs, rhs);
  }

  /**
   * @brief Solve the linear system. A x = y. (Matrix version)
   * 
   */
  MATHPRIM_NOINLINE void vsolve(matrix_view lhs, const_matrix_view rhs) {
    MATHPRIM_INTERNAL_CHECK_THROW(mat_.cols() == lhs.shape(0), std::runtime_error, "Invalid size of lhs");
    MATHPRIM_INTERNAL_CHECK_THROW(mat_.rows() == rhs.shape(0), std::runtime_error, "Invalid size of rhs");
    MATHPRIM_INTERNAL_CHECK_THROW(lhs.shape(1) == rhs.shape(1), std::runtime_error, "Invalid size of rhs");
    static_cast<Derived*>(this)->vsolve_impl(lhs, rhs);
  }

  /**
   * @brief Analyze the non-zero pattern of matrix.
   * 
   * @param mat 
   */
  MATHPRIM_NOINLINE Derived& analyze(const const_sparse& mat) {
    mat_ = mat;
    static_cast<Derived*>(this)->analyze_impl();
    return *static_cast<Derived*>(this);
  }

  MATHPRIM_NOINLINE Derived& factorize() {
    static_cast<Derived*>(this)->factorize_impl();
    return *static_cast<Derived*>(this);
  }

  MATHPRIM_NOINLINE Derived& compute(const const_sparse& mat) {
    return analyze(mat).factorize();
  }

protected:
  explicit basic_direct_solver(const_sparse mat) : mat_(mat) {}
  const_sparse mat_;
};

}