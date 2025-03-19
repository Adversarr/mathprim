#pragma once
#include "mathprim/linalg/iterative/iterative.hpp"

namespace mathprim::sparse::iterative {

/**
 * @brief Smoother base, it approximately solves the linear system.
 *
 * @tparam Derived
 * @tparam Scalar
 * @tparam Device
 */
template <typename Derived, typename Scalar, typename Device, sparse_format Format>
class basic_smoother {
public:
  using vector_type = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  using const_vector = contiguous_view<const Scalar, shape_t<keep_dim>, Device>;
  using const_sparse = basic_sparse_view<const Scalar, Device, Format>;
  basic_smoother() = default;
  explicit basic_smoother(const_sparse mat): mat_(mat) {}

  Derived& derived() noexcept { return *static_cast<Derived*>(this); }
  const Derived& derived() const noexcept { return *static_cast<const Derived*>(this); }

  Derived& analyze(const_sparse mat) {
    if (mat) {
      mat_ = mat;
    }
    MATHPRIM_INTERNAL_CHECK_THROW(mat_, std::runtime_error, "No matrix provided.");
    derived().analyze_impl();
    return derived();
  }

  Derived& factorize() {
    MATHPRIM_INTERNAL_CHECK_THROW(mat_, std::runtime_error, "No matrix provided.");
    derived().factorize_impl();
    return derived();
  }

  Derived& compute(const_sparse mat) { return analyze(mat).factorize(); }

  // A x = b
  void apply(vector_type dx, const_vector residual) {
    auto rows = mat_.rows();
    MATHPRIM_INTERNAL_CHECK_THROW(dx.size() == rows, shape_error, "The size of dx is not equal to the number of rows.");
    MATHPRIM_INTERNAL_CHECK_THROW(residual.size() == rows, shape_error,
                                  "The size of residual is not equal to the number of rows.");
    static_cast<Derived*>(this)->apply_impl(dx, residual);
  }

  const_sparse matrix() const noexcept { return mat_; }

protected:
  const_sparse mat_;
};

}  // namespace mathprim::sparse::iterative