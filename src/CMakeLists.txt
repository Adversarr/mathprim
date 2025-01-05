add_library(mathprim INTERFACE)
target_include_directories(mathprim INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

if(MATHPRIM_ENABLE_WARNINGS)
  target_compile_options(mathprim INTERFACE -Wall)
endif()

if (MATHPRIM_ENABLE_EIGEN)
  find_package(Eigen3 REQUIRED)
  message(STATUS "Enabling Eigen")
  target_link_libraries(mathprim INTERFACE Eigen3::Eigen)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_EIGEN)
endif()

if (MATHPRIM_ENABLE_BLAS)
  find_package(BLAS REQUIRED)
  message(STATUS "Enabling BLAS")

  target_link_libraries(mathprim INTERFACE ${BLAS_LIBRARIES})
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_BLAS)
endif()

if (MATHPRIM_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Enabling CUDA")

  target_link_libraries(mathprim INTERFACE CUDA::cudart CUDA::cublas)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_CUDA)
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()
