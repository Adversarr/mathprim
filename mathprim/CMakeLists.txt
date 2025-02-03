add_library(mathprim INTERFACE)
target_include_directories(mathprim INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

if(MATHPRIM_ENABLE_WARNINGS)
if(MSVC)
  target_compile_options(mathprim INTERFACE /W4)
else()
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>)
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()
endif()

if(MATHPRIM_ENABLE_EIGEN)
  find_package(Eigen3 REQUIRED)
  message(STATUS "Enabling Eigen")
  target_link_libraries(mathprim INTERFACE Eigen3::Eigen)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_EIGEN)
endif()

if(MATHPRIM_ENABLE_BLAS)
  if (APPLE)
    # Prefer to use Accelerate framework on macOS
    set(BLA_VENDOR "Apple")
    target_compile_definitions(mathprim INTERFACE MATHPRIM_BLAS_VENDOR_APPLE)
    target_compile_definitions(mathprim INTERFACE ACCELERATE_NEW_LAPACK)
  else()
    set(BLA_VENDOR "OpenBLAS")
    target_compile_definitions(mathprim INTERFACE MATHPRIM_BLAS_VENDOR_OPENBLAS)
  endif()

  find_package(BLAS REQUIRED)
  message(STATUS "Enabling BLAS: ${BLAS_LIBRARIES} ${BLA_VENDOR}")

  target_link_libraries(mathprim INTERFACE ${BLAS_LIBRARIES})
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_BLAS)
endif()

if(MATHPRIM_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Enabling CUDA")

  target_link_libraries(mathprim INTERFACE CUDA::cudart CUDA::cublas)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_CUDA)
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

if(MATHPRIM_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  message(STATUS "Enabling OpenMP")

  target_link_libraries(mathprim INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_OPENMP)
endif()
