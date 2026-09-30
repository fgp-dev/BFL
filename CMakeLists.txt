# Copyright 2026 Facundo Gomez Prates
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(bfl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(bfl STATIC src/bfl.cpp src/greedy.cpp)
target_compile_features(bfl PUBLIC cxx_std_17)
target_include_directories(bfl PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                       $<INSTALL_INTERFACE:include>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bfl PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(bfl_example examples/minimal.cpp)
target_link_libraries(bfl_example PRIVATE bfl)

add_executable(bfl_tests tests/test_bfl.cpp)
target_link_libraries(bfl_tests PRIVATE bfl)
add_executable(bfl_greedy_tests tests/test_greedy.cpp)
target_link_libraries(bfl_greedy_tests PRIVATE bfl)
enable_testing()
add_test(NAME bfl_tests COMMAND bfl_tests)
add_test(NAME bfl_greedy_tests COMMAND bfl_greedy_tests)

add_executable(bfl_benchmark benchmarks/benchmark.cpp)
target_link_libraries(bfl_benchmark PRIVATE bfl)
add_executable(bfl_generalization benchmarks/generalization.cpp)
target_link_libraries(bfl_generalization PRIVATE bfl)

include(GNUInstallDirs)
install(TARGETS bfl EXPORT bflTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT bflTargets NAMESPACE bfl:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bfl)
