#pragma once
#include "mathprim/core/view.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::sparse::direct {

template <typename Derived, typename Scalar, sparse_format Format, typename Device>
class basic_direct_solver {
public:
  using vector_view = contiguous_view<Scalar, dshape<1>, Device>;
  using const_vector = contiguous_view<const Scalar, dshape<1>, Device>;

  using matrix_view = basic_sparse_view<Scalar, Device, Format>;
  using const_matrix_view = basic_sparse_view<const Scalar, Device, Format>;

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

  MATHPRIM_NOINLINE Derived& analyze(const const_matrix_view& mat) {
    mat_ = mat;
    static_cast<Derived*>(this)->analyze_impl();
    return *static_cast<Derived*>(this);
  }

  MATHPRIM_NOINLINE Derived& factorize() {
    static_cast<Derived*>(this)->factorize_impl();
    return *static_cast<Derived*>(this);
  }

  MATHPRIM_NOINLINE Derived& compute(const const_matrix_view& mat) {
    return analyze(mat).factorize();
  }

protected:
  explicit basic_direct_solver(const_matrix_view mat) : mat_(mat) {}
  const_matrix_view mat_;
};

}