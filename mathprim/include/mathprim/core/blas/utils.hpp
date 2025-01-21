#pragma once
#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"
namespace mathprim::blas::internal {

void check_mm_shapes(const dim<2>& a, const dim<2>& b, const dim<2>& c) {
  if (a[1] != b[0]) {
    throw shape_error("blas::gemm: A.shape(1) != B.shape(0)");
  } else if (a[0] != c[0]) {
    throw shape_error("blas::gemm: A.shape(0) != C.shape(0)");
  } else if (b[1] != c[1]) {
    throw shape_error("blas::gemm: B.shape(1) != C.shape(1)");
  }
}

void check_mv_shapes(const dim<2>& a, const dim<1>& x, const dim<1>& y) {
  if (a[1] != x[0]) {
    throw shape_error("blas::gemv: A.shape(1) != x.shape(0)");
  } else if (a[0] != y[0]) {
    throw shape_error("blas::gemv: A.shape(0) != y.shape(0)");
  }
}

}  // namespace mathprim::blas::internal
