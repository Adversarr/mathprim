find_package(GTest REQUIRED)
enable_testing()

add_subdirectory(dim_iterator)
add_subdirectory(cpu_blas)
add_subdirectory(hello_world)
add_subdirectory(omp)
add_subdirectory(benchmark)
add_subdirectory(sparse_csr)
add_subdirectory(cg_solver)
add_subdirectory(npyio)

if(MATHPRIM_BUILD_TESTS_CUDA)
  add_subdirectory(hello_world_cuda)
  add_subdirectory(cublas)
  add_subdirectory(sparse)
  add_subdirectory(cuda_training_series)

  add_subdirectory(eigen_cuda)
endif()

add_subdirectory(eigen)
add_subdirectory(nystorm)
add_subdirectory(cholmod)