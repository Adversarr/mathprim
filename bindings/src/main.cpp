#include "checkings.hpp"
#include "cholesky.hpp"
#include "geometry.hpp"
#include "linalg.hpp"

bool is_cuda_available() {
#ifdef MATHPRIM_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

NB_MODULE(libpymp, m) {
  ////////// Basic //////////
  m.doc() = "MathPrim: A lightweight tensor(view) library";
  m.attr("__version__") = "0.1.0";
  auto checking = m.def_submodule("checking", "Checking module (internal debug use).");
  bind_checkings(checking);

  ////////// Geometry //////////
  auto geometry = m.def_submodule("geometry", "Geometry module, including mesh, laplacian, mass, etc.");
  bind_geometry(geometry);

  ////////// Linalg //////////
  auto linalg = m.def_submodule("linalg", "Linear algebra module, including matrix, vector, etc.");
  bind_linalg(linalg);
  bind_linalg_cuda(linalg);
  
  ////////// Cholmod //////////
  bind_cholesky(linalg);

  ////////// CUDA //////////
  m.def("is_cuda_available", &is_cuda_available, "Check if CUDA is available.");
}
