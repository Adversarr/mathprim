#pragma once
#include "mathprim/core/view.hpp"
#include "mathprim/supports/eigen_dense.hpp"

namespace mathprim::linalg {

/**
 * @brief Ensure that the input matrix is symmetric positive semi-definite enough, such that the smallest eigenvalue is
 * at least eps. This method assumes the input matrix is symmetric.
 *
 * @tparam Scalar
 * @tparam Device
 * @tparam Nrows
 */
template <typename Scalar, typename Device, index_t Nrows>
struct make_spsd {
  using const_matrix = contiguous_view<const Scalar, shape_t<Nrows, Nrows>, Device>;
  using matrix_type = contiguous_view<Scalar, shape_t<Nrows, Nrows>, Device>;
  using work_matrix = Eigen::Matrix<Scalar, Nrows, Nrows>;
  using work_vector = Eigen::Vector<Scalar, Nrows>;
  using work_solver = Eigen::SelfAdjointEigenSolver<work_matrix>;

  MATHPRIM_PRIMFUNC explicit make_spsd(Scalar eps = std::numeric_limits<Scalar>::epsilon()) noexcept : eps_(eps) {}
  make_spsd(const make_spsd&) = default;
  make_spsd(make_spsd&&) = default;
  make_spsd& operator=(const make_spsd&) = default;
  make_spsd& operator=(make_spsd&&) = default;

  MATHPRIM_PRIMFUNC void operator()(matrix_type m) const noexcept {
    auto mapped = eigen_support::cmap(m);
    work_matrix temp = mapped;
    work_solver solver(temp);
    work_vector eigvals = solver.eigenvalues();
    eigvals = eigvals.cwiseMax(eps_);
    temp.noalias() = solver.eigenvectors() * eigvals.asDiagonal() * solver.eigenvectors().transpose();
    mapped = temp;
  }

  Scalar eps_;
};

/**
 * @brief Computes the eigenvalues and eigenvectors of a real symmetric matrix.
 *
 * @tparam Scalar
 * @tparam Device
 * @tparam Nrows
 */
template <typename Scalar, typename Device, index_t Nrows>
struct eigh {
  using const_matrix = contiguous_view<const Scalar, shape_t<Nrows, Nrows>, Device>;
  using matrix_type = contiguous_view<Scalar, shape_t<Nrows, Nrows>, Device>;
  using vector_type = contiguous_view<Scalar, shape_t<Nrows>, Device>;
  using work_matrix = Eigen::Matrix<Scalar, Nrows, Nrows>;
  using work_vector = Eigen::Vector<Scalar, Nrows>;
  using work_solver = Eigen::SelfAdjointEigenSolver<work_matrix>;

  MATHPRIM_PRIMFUNC void operator() (const_matrix m, matrix_type eigvec, vector_type eigval) const noexcept {
    auto mapped = eigen_support::cmap(m);
    work_matrix temp = mapped;
    work_solver solver(temp);
    eigen_support::cmap(eigvec).transpose() = solver.eigenvectors();
    eigen_support::cmap(eigval) = solver.eigenvalues();
  }
};

}