mathprim_add_example(optim_quad quad.cpp)
mathprim_add_example(optim_banana banana.cpp)

if(MATHPRIM_BUILD_TESTS_CUDA)
  mathprim_add_example(cu_optim_quad quad.cu)
endif(MATHPRIM_BUILD_TESTS_CUDA)