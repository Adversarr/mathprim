find_package(Python COMPONENTS Interpreter Development REQUIRED)
file(GLOB_RECURSE MATHPRIM_BINDINGS bindings/**.cpp)
nanobind_add_module(pymathprim  ${MATHPRIM_BINDINGS})
target_link_libraries(pymathprim PRIVATE mathprim)