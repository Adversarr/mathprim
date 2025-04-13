#include "linalg.hpp"

#include <nanobind/stl/function.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <mathprim/blas/cpu_blas.hpp>
#include <mathprim/linalg/iterative/precond/ainv.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/precond/eigen_support.hpp>
#include <mathprim/linalg/iterative/precond/fsai0.hpp>
#include <mathprim/linalg/iterative/precond/sparse_inverse.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/sparse/basic_sparse.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/sparse/systems/laplace.hpp>

#include "mathprim/sparse/blas/naive.hpp"

using namespace mathprim;
using namespace helper;

template <typename Flt, typename Precond, typename Callback>
static std::tuple<index_t, double, double>                    //
solve_cg(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,  //
         nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,    //
         nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,    //
         const Flt rtol, index_t max_iter, Callback callback) {
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
  auto view_a = eigen_support::view(A);
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  using Blas = mp::blas::cpu_blas<Flt>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cpu, SparseBlas, Blas, Precond>;
  auto start = helper::time_now();
  Solver solver(view_a);
  double prec = time_elapsed(start);
  sparse::convergence_criteria<Flt> criteria{max_iter, rtol};
  start = std::chrono::high_resolution_clock::now();
  sparse::convergence_result<Flt> result;
  result = solver.solve(x_view, b_view, criteria, callback);
  double solve = time_elapsed(start);
  return {result.iterations_, prec, solve};
}

template <typename Flt, typename Precond>
static std::tuple<index_t, double, double>                    //
solve_cg(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,  //
         nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,    //
         nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,    //
         const Flt rtol, index_t max_iter) {
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
  auto view_a = eigen_support::view(A);
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  using Blas = mp::blas::cpu_blas<Flt>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cpu, SparseBlas, Blas, Precond>;
  auto start = helper::time_now();
  Solver solver(view_a);
  double prec = time_elapsed(start);
  sparse::convergence_criteria<Flt> criteria{max_iter, rtol};
  start = std::chrono::high_resolution_clock::now();
  sparse::convergence_result<Flt> result;
  result = solver.solve(x_view, b_view, criteria);
  double solve = time_elapsed(start);
  return {result.iterations_, prec, solve};
}

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
static std::tuple<index_t, double, double>                   //
cg_host(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,  //
        nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,    //
        nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,    //
        const Flt rtol,                                      //
        index_t max_iter,                                    //
        int verbose) {
  if (verbose > 0) {
    // verbose function is print
    return solve_cg<Flt, Precond>(A, b, x, rtol, max_iter, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    return solve_cg<Flt, Precond>(A, b, x, rtol, max_iter);
  }
}

template <typename Flt, typename Precond = sparse::iterative::none_preconditioner<Flt, device::cpu>>
static std::tuple<index_t, double, double> cg_host_callback(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,  //
                                                            nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,    //
                                                            nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,    //
                                                            const Flt rtol,                                      //
                                                            index_t max_iter,                                    //
                                                            std::function<void(index_t, Flt)> callback) {
  return solve_cg<Flt, Precond>(A, b, x, rtol, max_iter, callback);
}

template <typename Scalar>
using diagonal = sparse::iterative::diagonal_preconditioner<Scalar, device::cpu, sparse::sparse_format::csr,
                                                            blas::cpu_blas<Scalar>>;

template <typename Scalar>
using no = sparse::iterative::none_preconditioner<Scalar, device::cpu, mathprim::sparse::sparse_format::csr>;

template <typename Scalar>
using ainv = sparse::iterative::bridson_ainv_preconditioner<sparse::blas::eigen<Scalar, sparse::sparse_format::csr>,
                                                            blas::cpu_blas<Scalar>>;

template <typename Scalar>
using ic = sparse::iterative::eigen_ichol<Scalar, sparse::sparse_format::csr>;

template <typename Flt = float>
Eigen::SparseMatrix<Flt, Eigen::RowMajor> ainv_content(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A) {
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  sparse::iterative::scaled_bridson_ainv_preconditioner<SparseBlas> ainv(eigen_support::view(A));
  return eigen_support::map(ainv.ainv());
}

template <typename Flt = float>
static Eigen::SparseMatrix<Flt, Eigen::RowMajor> ichol_content(  //
    Eigen::SparseMatrix<Flt, Eigen::RowMajor> m                  //
) {
  using namespace Eigen;
  IncompleteCholesky<Flt, Lower, NaturalOrdering<index_t>> ic;
  ic.compute(m);
  if (ic.info() != Success) {
    throw std::runtime_error("Incomplete Cholesky factorization failed.");
  }

  auto lower_mat = ic.matrixL();
  return lower_mat;
}
template <typename Flt = float>
std::tuple<index_t, double, double> pcg_with_ext_spai(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,     //
                                                      nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,       //
                                                      nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,       //
                                                      const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& ainv,  //
                                                      Flt epsilon,                                            //
                                                      const Flt& rtol,                                        //
                                                      index_t max_iter,                                       //
                                                      int verbose) {
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  using Blas = mp::blas::cpu_blas<Flt>;
  using Precond = mp::sparse::iterative::sparse_preconditioner<SparseBlas, Blas>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cpu, SparseBlas, Blas, Precond>;
  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  auto b_view = view(b.data(), make_shape(b.size()));
  auto x_view = view(x.data(), make_shape(x.size()));
  if (b_view.size() != A.rows() || x_view.size() != A.cols()) {
    throw std::invalid_argument("b and x must have the same size as the matrix.");
  }

  auto start = helper::time_now();
  Solver solver(eigen_support::view(A));
  auto prec = time_elapsed(start);
  solver.preconditioner().derived().set_approximation(eigen_support::view(ainv), epsilon);

  sparse::convergence_criteria<Flt> criteria{
    max_iter,
    rtol,
  };
  sparse::convergence_result<Flt> result;
  if (verbose > 0) {
    result = solver.solve(x_view, b_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.solve(x_view, b_view, criteria);
  }
  double solve = time_elapsed(start);
  return {result.iterations_, prec, solve};
}

template <typename Flt = float>
std::tuple<index_t, double, double> pcg_with_ext_spai_callback(  //
    const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,          //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,            //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,            //
    const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& ainv,       //
    Flt epsilon,                                                 //
    const Flt& rtol,                                             //
    index_t max_iter,                                            //
    std::function<void(index_t, Flt)> callback) {
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  using Blas = mp::blas::cpu_blas<Flt>;
  using Precond = mp::sparse::iterative::sparse_preconditioner<SparseBlas, Blas>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cpu, SparseBlas, Blas, Precond>;
  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  auto b_view = view(b.data(), make_shape(b.size()));
  auto x_view = view(x.data(), make_shape(x.size()));
  if (b_view.size() != A.rows() || x_view.size() != A.cols()) {
    throw std::invalid_argument("b and x must have the same size as the matrix.");
  }

  auto start = helper::time_now();
  Solver solver(eigen_support::view(A));
  auto prec = time_elapsed(start);
  // solver.preconditioner().derived().set_approximation(eigen_support::view(ainv), epsilon);
  solver.preconditioner().derived().set_approximation(ainv, epsilon);

  sparse::convergence_criteria<Flt> criteria{
    max_iter,
    rtol,
  };
  auto result = solver.solve(x_view, b_view, criteria, callback);
  double solve = time_elapsed(start);
  return {result.iterations_, prec, solve};
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

  m.def("pcg_cb_with_ext_spai", &pcg_with_ext_spai_callback<double>,                   //
        "Preconditioned Conjugate Gradient method on CPU.",                            //
        nb::arg("A").noconvert(), nb::arg("b").noconvert(), nb::arg("x").noconvert(),  //
        nb::arg("ainv").noconvert(), nb::arg("epsilon"),                               //
        nb::arg("rtol"), nb::arg("max_iter"), nb::arg("callback"));
  m.def("pcg_cb_with_ext_spai", &pcg_with_ext_spai_callback<float>,                    //
        "Preconditioned Conjugate Gradient method on CPU.",                            //
        nb::arg("A").noconvert(), nb::arg("b").noconvert(), nb::arg("x").noconvert(),  //
        nb::arg("ainv").noconvert(), nb::arg("epsilon"),                               //
        nb::arg("rtol"), nb::arg("max_iter"), nb::arg("callback"));

  ////////// Helpers for Preconditioners //////////
  m.def("ainv_content", &ainv_content<float>, "Get the content of the Approx Inverse Preconditioner.",
        nb::arg("A").noconvert());
  m.def("ainv_content", &ainv_content<double>, "Get the content of the Approx Inverse Preconditioner.",
        nb::arg("A").noconvert());
  m.def("ichol_content", &ichol_content<double>, "Compute the incomplete Cholesky factorization of a matrix.",  //
        nb::arg("m").noconvert());
  m.def("ichol_content", &ichol_content<float>, "Compute the incomplete Cholesky factorization of a matrix.",  //
        nb::arg("m").noconvert());

  ////////// Model Problems //////////
  m.def("grid_laplacian_nd_dbc_float32", &grid_laplacian_nd_dbc<float>,  //
        "Grid Laplacian matrix with Dirichlet BCs.", nb::arg("dims"));
  m.def("grid_laplacian_nd_dbc_float64", &grid_laplacian_nd_dbc<double>,  //
        "Grid Laplacian matrix with Dirichlet BCs.", nb::arg("dims"));
}
#ifndef MATHPRIM_ENABLE_CUDA
void bind_linalg_cuda(nb::module_& m) {
  // Do nothing
}
#endif
