#ifndef MATHPRIM_BACKEND_BLAS_HELPER_MACROS
#define MATHPRIM_BACKEND_BLAS_HELPER_MACROS

#define MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device) \
  backend_blas<Type, Blas, Device>

#define MATHPRIM_BACKEND_BLAS_COPY_IMPL(Type, Blas, Device)            \
  void MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::copy(           \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::vector_view dst, \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_vector_view src)

#define MATHPRIM_BACKEND_BLAS_SCAL_IMPL(Type, Blas, Device)  \
  void MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::scal( \
      Type alpha,                                            \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::vector_view x)

#define MATHPRIM_BACKEND_BLAS_SWAP_IMPL(Type, Blas, Device)          \
  void MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::swap(         \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::vector_view x, \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::vector_view y)

#define MATHPRIM_BACKEND_BLAS_AXPY_IMPL(Type, Blas, Device)                \
  void MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::axpy(               \
      Type alpha,                                                          \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_vector_view x, \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::vector_view y)

#define MATHPRIM_BACKEND_BLAS_DOT_IMPL(Type, Blas, Device)                 \
  Type MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::dot(                \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_vector_view x, \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_vector_view y)

#define MATHPRIM_BACKEND_BLAS_NORM_IMPL(Type, Blas, Device)  \
  Type MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::norm( \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_vector_view x)

#define MATHPRIM_BACKEND_BLAS_ASUM_IMPL(Type, Blas, Device)  \
  Type MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::asum( \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_vector_view x)

#define MATHPRIM_BACKEND_BLAS_AMAX_IMPL(Type, Blas, Device)  \
  Type MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::amax( \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_vector_view x)

#define MATHPRIM_BACKEND_BLAS_GEMV_IMPL(Type, Blas, Device)                \
  void MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::gemv(               \
      Type alpha,                                                          \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_matrix_view A, \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_vector_view x, \
      Type beta,                                                           \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::vector_view y)

#define MATHPRIM_BACKEND_BLAS_GEMM_IMPL(Type, Blas, Device)                \
  void MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::gemm(               \
      Type alpha,                                                          \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_matrix_view A, \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::const_matrix_view B, \
      Type beta,                                                           \
      MATHPRIM_BACKEND_BLAS_TYPE(Type, Blas, Device)::matrix_view C)

#endif
