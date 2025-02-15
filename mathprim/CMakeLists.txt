add_library(mathprim INTERFACE)
target_include_directories(mathprim INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(MATHPRIM_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
  else()
    target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>)
  endif()
endif()

if(MATHPRIM_ENABLE_EIGEN)
  find_package(Eigen3 REQUIRED)
  message(STATUS "Enabling Eigen")
  target_link_libraries(mathprim INTERFACE Eigen3::Eigen)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_EIGEN)
endif()

if(MATHPRIM_ENABLE_BLAS)
  if(APPLE)
    # Prefer to use Accelerate framework on macOS
    set(BLA_VENDOR "Apple")
    target_compile_definitions(mathprim INTERFACE MATHPRIM_BLAS_VENDOR_APPLE)
    target_compile_definitions(mathprim INTERFACE ACCELERATE_NEW_LAPACK)
  else()
    set(BLA_VENDOR "OpenBLAS")
    target_compile_definitions(mathprim INTERFACE MATHPRIM_BLAS_VENDOR_OPENBLAS)
  endif()

  find_package(BLAS REQUIRED)
  message(STATUS "Enabling BLAS: ${BLAS_LIBRARIES} ${BLA_VENDOR}")

  target_link_libraries(mathprim INTERFACE ${BLAS_LIBRARIES})
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_BLAS)
endif()

if(MATHPRIM_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Enabling CUDA")

  target_link_libraries(mathprim INTERFACE CUDA::cudart CUDA::cublas CUDA::cusparse)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_CUDA)
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  # OpenMP
  if (MATHPRIM_ENABLE_OPENMP)
    if (MSVC)
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler:/openmp>)
    else()
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    endif()
  endif()


  # Warnings:
  if(MATHPRIM_ENABLE_WARNINGS)

    # cross execution space call:
    if(MSVC)
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>)
    else()
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>)
    endif()
  endif()
endif()

if(MATHPRIM_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  message(STATUS "Enabling OpenMP")

  target_link_libraries(mathprim INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_OPENMP)
endif()

if(MATHPRIM_ENABLE_NANOBIND)
  add_subdirectory(nb)
endif()

if(MATHPRIM_ENABLE_CHOLMOD)
  find_package(CHOLMOD REQUIRED)
  message(STATUS "Enabling CHOLMOD")

  target_link_libraries(mathprim INTERFACE SuiteSparse::CHOLMOD)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_CHOLMOD)
endif()


include(GNUInstallDirs)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh")

install(TARGETS mathprim
  EXPORT mathprimTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT mathprimTargets
  FILE mathprimTargets.cmake
  NAMESPACE mathprim::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mathprim)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/../cmake/mathprimConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/mathprimConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mathprim
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/mathprimConfigVersion.cmake
  VERSION 0.0.1
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/mathprimConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/mathprimConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mathprim
)
