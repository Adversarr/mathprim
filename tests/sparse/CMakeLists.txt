add_executable(hello_sparse main.cu)
find_package(CUDAToolkit)
target_link_libraries(hello_sparse PRIVATE mathprim CUDA::cusparse)