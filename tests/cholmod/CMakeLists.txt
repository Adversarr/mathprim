find_package(benchmark REQUIRED)
add_executable(cholmod_chol main.cpp)
target_link_libraries(cholmod_chol PRIVATE mathprim benchmark::benchmark)