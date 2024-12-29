cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(
    FATAL_ERROR
    "In-source builds not allowed. Please make a new directory (called
      a build directory) and run CMake from there. You may need to
      remove CMakeCache.txt. "
  )
endif()

project(mathprim)

# master project detection with cmake variable
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MATHPRIM_MASTER_PROJECT ON)
else()
  set(MATHPRIM_MASTER_PROJECT OFF)
endif()

add_subdirectory(src)

option(MATHPRIM_BUILD_TESTS "Build tests" ${MATHPRIM_MASTER_PROJECT})

if(MATHPRIM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

