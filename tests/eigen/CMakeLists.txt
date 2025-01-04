find_package(Eigen3 REQUIRED)

add_executable(eigen main.cpp)
target_link_libraries(eigen PRIVATE Eigen3::Eigen mathprim)