#include "cholesky.hpp"
#ifdef MATHPRIM_ENABLE_CHOLMOD
#  include <mathprim/linalg/direct/cholmod.hpp>
#endif
#include <mathprim/linalg/direct/eigen_support.hpp>

using namespace mp;

template <typename Flt, template <typename, sparse::sparse_format> class Solver>
class direct_solver_interface {
public:
  explicit direct_solver_interface(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A) : alg_(eigen_support::view(A)) {}

  /// vector version: A x = b
  void solve(nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x, nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b) {
    auto x_view = nbex::to_mathprim(x);
    auto b_view = nbex::to_mathprim(b);
    alg_.solve(x_view, b_view);
  }

  /// matrix version: A X = B
  void vsolve(nb::ndarray<Flt, nb::ndim<2>, nb::device::cpu, nb::c_contig> X,
              nb::ndarray<Flt, nb::ndim<2>, nb::device::cpu, nb::c_contig> B) {
    auto x_view = nbex::to_mathprim(X);
    auto b_view = nbex::to_mathprim(B);
    alg_.vsolve(x_view, b_view);
  }

private:
  Solver<Flt, sparse::sparse_format::csr> alg_;
};

template <typename Flt, template <typename, sparse::sparse_format> class Solver>
static void bind_class(nb::module_& m, const std::string& name) {
  nb::class_<direct_solver_interface<Flt, Solver>>(m, name.c_str())
      .def(nb::init<const Eigen::SparseMatrix<Flt, Eigen::RowMajor>&>())
      .def("solve", &direct_solver_interface<Flt, Solver>::solve, nb::arg("x").noconvert(), nb::arg("b").noconvert())
      .def("vsolve", &direct_solver_interface<Flt, Solver>::vsolve, nb::arg("X").noconvert(), nb::arg("B").noconvert());
}

#ifdef MATHPRIM_ENABLE_CHOLMOD
static void bind_cholmod(nb::module_& m) {
  bind_class<float, sparse::direct::cholmod_chol>(m, "cholmod_cholesky_float32");
  bind_class<double, sparse::direct::cholmod_chol>(m, "cholmod_cholesky_float64");
}
#endif

void bind_cholesky(nb::module_& linalg) {
  auto cholmod = linalg.def_submodule("cholmod", "Cholmod module, including Cholesky decomposition.");
#ifdef MATHPRIM_ENABLE_CHOLMOD
  linalg.attr("is_cholmod_available") = true;
  bind_cholmod(cholmod);
#else
  linalg.attr("is_cholmod_available") = false;
#endif
  bind_class<float, sparse::direct::eigen_simplicial_ldlt>(linalg, "eigen_simplicial_ldlt_float32");
  bind_class<double, sparse::direct::eigen_simplicial_ldlt>(linalg, "eigen_simplicial_ldlt_float64");
  bind_class<float, sparse::direct::eigen_simplicial_llt>(linalg, "eigen_simplicial_llt_float32");
  bind_class<double, sparse::direct::eigen_simplicial_llt>(linalg, "eigen_simplicial_llt_float64");
}