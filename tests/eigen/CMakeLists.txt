find_package(Eigen3 REQUIRED)

add_executable(eigen main.cpp)
target_link_libraries(eigen PRIVATE mathprim GTest::GTest GTest::Main)
add_test(NAME eigen COMMAND eigen)