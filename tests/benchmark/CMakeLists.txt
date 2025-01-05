find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
add_executable(BM_map map.cpp)
target_link_libraries(BM_map PRIVATE benchmark::benchmark mathprim Eigen3::Eigen)

add_executable(BM_blas blas.cpp)
target_link_libraries(BM_blas PRIVATE benchmark::benchmark mathprim Eigen3::Eigen BLAS::BLAS)

add_executable(BM_par par.cpp)
target_link_libraries(BM_par PRIVATE benchmark::benchmark mathprim)