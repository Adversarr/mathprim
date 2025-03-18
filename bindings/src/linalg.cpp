#include "linalg.hpp"

#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <mathprim/blas/cpu_blas.hpp>
#include <mathprim/linalg/iterative/precond/approx_inv.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/precond/eigen_support.hpp>
#include <mathprim/linalg/iterative/precond/sparse_inverse.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/sparse/basic_sparse.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/sparse/systems/laplace.hpp>

using namespace mathprim;

////////////////////////////////////////////////
/// Model Problems
////////////////////////////////////////////////
template <typename Flt>
Eigen::SparseMatrix<Flt, Eigen::RowMajor> grid_laplacian_nd_dbc(std::vector<index_t> dims) {
  if (dims.size() == 0) {
    throw std::invalid_argument("dims must be non-empty.");
  }

  for (auto d : dims) {
    if (d < 2) {
      throw std::invalid_argument("dims must be at least 2.");
    }
  }

  mp::sparse::basic_sparse_matrix<Flt, mp::device::cpu, sparse::sparse_format::csr> res;
  if (dims.size() == 1) {
    mp::sparse::laplace_operator<Flt, 1> lop(make_shape(dims[0]));
    res = lop.template matrix<mp::sparse::sparse_format::csr>();
  } else if (dims.size() == 2) {
    mp::sparse::laplace_operator<Flt, 2> lop(make_shape(dims[0], dims[1]));
    res = lop.template matrix<mp::sparse::sparse_format::csr>();
  } else if (dims.size() == 3) {
    mp::sparse::laplace_operator<Flt, 3> lop(make_shape(dims[0], dims[1], dims[2]));
    res = lop.template matrix<mp::sparse::sparse_format::csr>();
  } else {
    throw std::invalid_argument("dims must be at most 3.");
  }

  return eigen_support::map(res.view());
}

////////////////////////////////////////////////
/// Conjugate Gradient Solvers.
////////////////////////////////////////////////
template <typename Flt, typename Precond = sparse::iterative::none_preconditioner<Flt, device::cpu>>
static std::pair<index_t, double> cg_host(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,  //
                                          nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,    //
                                          nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,    //
                                          const Flt rtol,                                      //
                                          index_t max_iter,                                    //
                                          int verbose) {
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = mp::blas::cpu_blas<Flt>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cpu, LinearOp, Blas, Precond>;

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  auto b_view = view(b.data(), make_shape(b.size()));
  auto x_view = view(x.data(), make_shape(x.size()));
  if (b_view.size() != A.rows() || x_view.size() != A.cols()) {
    throw std::invalid_argument("b and x must have the same size as the matrix.");
  }
  if (max_iter == 0) {
    max_iter = A.rows();
  }
  auto view_A = eigen_support::view(A);
  Solver solver(LinearOp(view_A), Blas{}, Precond{view_A});
  sparse::iterative::convergence_criteria<Flt> criteria{
    .max_iterations_ = max_iter,
    .norm_tol_ = rtol,
  };
  sparse::iterative::convergence_result<Flt> result;
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose > 0) {
    result = solver.apply(b_view, x_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.apply(b_view, x_view, criteria);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(duration).count();
  return std::make_pair(result.iterations_, seconds);
}

template <typename Flt, typename Precond = sparse::iterative::none_preconditioner<Flt, device::cpu>>
static std::pair<index_t, double> cg_host_callback(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,  //
                                                   nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,    //
                                                   nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,    //
                                                   const Flt rtol,                                      //
                                                   index_t max_iter,                                    //
                                                   std::function<void(index_t, Flt)> callback) {
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = mp::blas::cpu_blas<Flt>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cpu, LinearOp, Blas, Precond>;

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  auto b_view = view(b.data(), make_shape(b.size()));
  auto x_view = view(x.data(), make_shape(x.size()));
  if (b_view.size() != A.rows() || x_view.size() != A.cols()) {
    throw std::invalid_argument("b and x must have the same size as the matrix.");
  }
  if (max_iter == 0) {
    max_iter = A.rows();
  }
  auto view_A = eigen_support::view(A);
  Solver solver(LinearOp(view_A), Blas{}, Precond{view_A});
  sparse::iterative::convergence_criteria<Flt> criteria{
    .max_iterations_ = max_iter,
    .norm_tol_ = rtol,
  };
  sparse::iterative::convergence_result<Flt> result;
  auto start = std::chrono::high_resolution_clock::now();
  result = solver.apply(b_view, x_view, criteria, callback);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(duration).count();
  return std::make_pair(result.iterations_, seconds);
}

template <typename Scalar>
using diagonal = sparse::iterative::diagonal_preconditioner<Scalar, device::cpu, sparse::sparse_format::csr,
                                                            blas::cpu_blas<Scalar>>;

template <typename Scalar>
using no = sparse::iterative::none_preconditioner<Scalar, device::cpu>;

template <typename Scalar>
using ainv = sparse::iterative::approx_inverse_preconditioner<sparse::blas::eigen<Scalar, sparse::sparse_format::csr>>;

template <typename Scalar>
using ic = sparse::iterative::eigen_ichol<Scalar>;

template <typename Flt = float>
Eigen::SparseMatrix<Flt, Eigen::RowMajor> ainv_content(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A) {
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  sparse::iterative::approx_inverse_preconditioner<SparseBlas> ainv(eigen_support::view(A));

  return eigen_support::map(ainv.ainv());
}

template <typename Flt = float>
std::pair<index_t, double> pcg_with_ext_spai(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,     //
                                             nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,       //
                                             nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,       //
                                             const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& ainv,  //
                                             Flt epsilon,                                            //
                                             const Flt& rtol,                                        //
                                             index_t max_iter,                                       //
                                             int verbose) {
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = mp::blas::cpu_blas<Flt>;
  using Precond = mp::sparse::iterative::sparse_preconditioner<SparseBlas, Blas>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cpu, LinearOp, Blas, Precond>;
  Solver solver(LinearOp(eigen_support::view(A)), Blas{}, Precond(eigen_support::view(ainv), epsilon));

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  auto b_view = view(b.data(), make_shape(b.size()));
  auto x_view = view(x.data(), make_shape(x.size()));
  if (b_view.size() != A.rows() || x_view.size() != A.cols()) {
    throw std::invalid_argument("b and x must have the same size as the matrix.");
  }

  sparse::iterative::convergence_criteria<Flt> criteria{
    .max_iterations_ = max_iter,
    .norm_tol_ = rtol,
  };
  sparse::iterative::convergence_result<Flt> result;
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose > 0) {
    result = solver.apply(b_view, x_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.apply(b_view, x_view, criteria);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(duration).count();
  return std::make_pair(result.iterations_, seconds);
}

#define BIND_TYPE(flt, preconditioning)                                                                     \
  m.def(TOSTR(pcg_##preconditioning), &cg_host<flt, preconditioning<flt>>,                                  \
        "Preconditioned CG on CPU (with " #preconditioning " precond.)", nb::arg("A").noconvert(),          \
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), nb::arg("rtol"), nb::arg("max_iter"),           \
        nb::arg("verbose") = 0);                                                                            \
  m.def(TOSTR(pcg_cb_##preconditioning), &cg_host_callback<flt, preconditioning<flt>>,                      \
        "Preconditioned CG on CPU+Callback (with " #preconditioning " precond.)", nb::arg("A").noconvert(), \
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), nb::arg("rtol"), nb::arg("max_iter"), nb::arg("fn"))

#define BIND_ALL(flt)       \
  BIND_TYPE(flt, no);       \
  BIND_TYPE(flt, diagonal); \
  BIND_TYPE(flt, ainv);     \
  BIND_TYPE(flt, ic)

void bind_linalg(nb::module_& m) {
  ////////// Solvers //////////
  BIND_ALL(float);
  BIND_ALL(double);

  ////////// External AI //////////
  m.def("pcg_with_ext_spai", &pcg_with_ext_spai<float>, "Preconditioned Conjugate Gradient method on CPU.",  //
        nb::arg("A").noconvert(), nb::arg("b").noconvert(), nb::arg("x").noconvert(),                        //
        nb::arg("ainv").noconvert(), nb::arg("epsilon"),                                                     //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
  m.def("pcg_with_ext_spai", &pcg_with_ext_spai<double>, "Preconditioned Conjugate Gradient method on CPU.",  //
        nb::arg("A").noconvert(), nb::arg("b").noconvert(), nb::arg("x").noconvert(),                         //
        nb::arg("ainv").noconvert(), nb::arg("epsilon"),                                                      //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);

  ////////// Helpers for Preconditioners //////////
  m.def("ainv_content", &ainv_content<float>, "Get the content of the Approx Inverse Preconditioner.",
        nb::arg("A").noconvert());
  m.def("ainv_content", &ainv_content<double>, "Get the content of the Approx Inverse Preconditioner.",
        nb::arg("A").noconvert());

  ////////// Model Problems //////////
  m.def("grid_laplacian_nd_dbc_float32", &grid_laplacian_nd_dbc<float>, "Grid Laplacian matrix with Dirichlet BCs.",
        nb::arg("dims"))
      .def("grid_laplacian_nd_dbc_float64", &grid_laplacian_nd_dbc<double>, "Grid Laplacian matrix with Dirichlet BCs.",
           nb::arg("dims"));
}
#ifndef MATHPRIM_ENABLE_CUDA
void bind_linalg_cuda(nb::module_& m) {
  // Do nothing
}
#endif
