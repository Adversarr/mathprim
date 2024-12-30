add_subdirectory(hello_world)
add_subdirectory(dim_iterator)

if(MATHPRIM_BUILD_TESTS_CUDA)
  add_subdirectory(hello_world_cuda)
endif()
