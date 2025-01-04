find_package(Eigen3 REQUIRED)

add_executable(eigen_cuda main.cu)
target_link_libraries(eigen_cuda PRIVATE Eigen3::Eigen mathprim)
target_compile_options(eigen_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
