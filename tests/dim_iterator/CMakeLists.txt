add_executable(dim_iterator main.cpp)
target_link_libraries(dim_iterator PRIVATE mathprim GTest::gtest_main)
add_test(NAME dim_iterator COMMAND dim_iterator)
