add_subdirectory(hello_world)

if(MATHPRIM_BUILD_TESTS_CUDA)
  add_subdirectory(hello_world_cuda)
endif()
