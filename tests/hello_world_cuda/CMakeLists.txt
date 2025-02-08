add_executable(hello_world_cuda main.cu)
target_link_libraries(hello_world_cuda PRIVATE mathprim)
