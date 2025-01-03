add_executable(hello_world_cuda main.cu)
target_link_libraries(hello_world_cuda PRIVATE mathprim)
target_compile_options(hello_world_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
