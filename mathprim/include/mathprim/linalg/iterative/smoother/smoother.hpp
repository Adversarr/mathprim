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
template <typename Derived, typename Scalar, typename Device>
class basic_smoother {
public:
  using vector_type = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  using const_vector = contiguous_view<const Scalar, shape_t<keep_dim>, Device>;

  explicit basic_smoother(index_t rows) : rows_(rows) {}

  void compute() {
    static_cast<Derived*>(this)->compute_impl();
  }

  // A x = b
  void apply(vector_type dx, const_vector residual) {
    // check shape
    MATHPRIM_INTERNAL_CHECK_THROW(dx.size() == rows_, shape_error,
                                  "The size of dx is not equal to the number of rows.");
    MATHPRIM_INTERNAL_CHECK_THROW(residual.size() == rows_, shape_error,
                                  "The size of residual is not equal to the number of rows.");

    // run
    static_cast<Derived*>(this)->apply_impl(dx, residual);
  }

  index_t rows() const noexcept {
    return rows_;
  }

protected:
  index_t rows_;
};
}  // namespace mathprim::sparse::iterative