find_package(GTest REQUIRED)
enable_testing()

add_subdirectory(dim_iterator)
add_subdirectory(cpu_blas)
add_subdirectory(hello_world)
add_subdirectory(omp)
add_subdirectory(benchmark)

if(MATHPRIM_BUILD_TESTS_CUDA)
  add_subdirectory(hello_world_cuda)
  add_subdirectory(cublas)
  add_subdirectory(cuda_training_series)

  if(MATHPRIM_ENABLE_EIGEN)
    add_subdirectory(eigen_cuda)
  endif()
endif()

if(MATHPRIM_ENABLE_EIGEN)
  add_subdirectory(eigen)
endif()
