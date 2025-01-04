find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)
add_executable(benchmark_cpu main.cpp)
target_link_libraries(benchmark_cpu PRIVATE benchmark::benchmark mathprim Eigen3::Eigen)