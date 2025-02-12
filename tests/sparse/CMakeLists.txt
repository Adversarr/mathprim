add_executable(cusparse main.cu)
find_package(CUDAToolkit)
target_link_libraries(cusparse PRIVATE mathprim CUDA::cusparse)