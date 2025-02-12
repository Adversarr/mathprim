add_executable(cg_solver main.cpp)
target_link_libraries(cg_solver PRIVATE mathprim)
add_test(NAME cg_solver COMMAND cg_solver)