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

# master project detection with cmake variable
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MATHPRIM_MASTER_PROJECT ON)
else()
  set(MATHPRIM_MASTER_PROJECT OFF)
endif()

project(mathprim)

# Test if cuda is available
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  set(MATHPRIM_ENABLE_CUDA_TEST ${MATHPRIM_MASTER_PROJECT})
else()
  set(MATHPRIM_ENABLE_CUDA_TEST OFF)
endif()

set(MATHPRIM_INTERNAL_ENABLE_BLAS ${MATHPRIM_MASTER_PROJECT})

find_package(CHOLMOD QUIET)
if(CHOLMOD_FOUND)
  set(MATHPRIM_INTERNAL_ENABLE_CHOLMOD ${MATHPRIM_MASTER_PROJECT})
else()
  set(MATHPRIM_INTERNAL_ENABLE_CHOLMOD OFF)
endif()

find_package(MKL QUIET)
if(MKL_FOUND)
  set(MATHPRIM_INTERNAL_ENABLE_MKL ${MATHPRIM_MASTER_PROJECT})
else()
  set(MATHPRIM_INTERNAL_ENABLE_MKL OFF)
endif()

option(MATHPRIM_ENABLE_OPENMP "Enable OpenMP" ON) # default to on, since most compilers support it
option(MATHPRIM_ENABLE_WARNINGS "Enable compiler warnings" ${MATHPRIM_MASTER_PROJECT})
option(MATHPRIM_ENABLE_BLAS "Enable BLAS" ${MATHPRIM_INTERNAL_ENABLE_BLAS})
option(MATHPRIM_ENABLE_CHOLMOD "Enable CHOLMOD" ${MATHPRIM_INTERNAL_ENABLE_CHOLMOD})
option(MATHPRIM_ENABLE_CUDA "Enable CUDA" ${MATHPRIM_ENABLE_CUDA_TEST})
option(MATHPRIM_ENABLE_MKL "Enable Intel MKL" ${MATHPRIM_INTERNAL_ENABLE_MKL})
option(MATHPRIM_BUILD_TESTS "Build tests" ${MATHPRIM_MASTER_PROJECT})
option(MATHPRIM_BUILD_TESTS_CUDA "Build cuda tests" ${MATHPRIM_ENABLE_CUDA_TEST})
option(MATHPRIM_INSTLALL "Install mathprim" OFF)
option(MATHPRIM_BUILD_BINDINGS "Build bindings" OFF)
option(MATHPRIM_CUDA_LINK_DYNAMIC "Link to dynamic CUDA libraries" OFF)

set(MATHPRIM_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR})

if(MATHPRIM_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_ARCHITECTURES 86)
  enable_language(CUDA)
endif()

add_subdirectory(mathprim)

if(MATHPRIM_BUILD_TESTS)
  enable_testing()
  message(STATUS "Building tests")
  add_subdirectory(tests)
endif()

if(MATHPRIM_BUILD_BINDINGS)
  set(PYMP_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR})
  add_subdirectory(bindings)
endif()