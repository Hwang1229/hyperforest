cmake_minimum_required(VERSION 3.13.4)

# hyperforest: a header-only, p4est-like parallel forest of 2^D-trees for
# any dimension D, with a deal.II adapter for phase-space problems.
#
#   core     include/hyperforest/{cell,connectivity,forest}.h   MPI, C++14
#   deal.II  include/hyperforest/dealii.h                        deal.II >= 9.5
#
# Use from another CMake project:
#   find_package(hyperforest REQUIRED)            # after installation, or
#   add_subdirectory(external/hyperforest)        # as a git submodule
#   target_link_libraries(my_target hyperforest::hyperforest)

project(hyperforest VERSION 0.1.0 LANGUAGES CXX)

option(HYPERFOREST_ENABLE_TESTS "Build the tests and the examples" ${PROJECT_IS_TOP_LEVEL})
option(HYPERFOREST_WITH_DEALII "Build the deal.II adapter tests and examples (needs deal.II)" ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_library(hyperforest INTERFACE)
add_library(hyperforest::hyperforest ALIAS hyperforest)
target_include_directories(hyperforest INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                 $<INSTALL_INTERFACE:include>)
target_compile_features(hyperforest INTERFACE cxx_std_14)
target_link_libraries(hyperforest INTERFACE MPI::MPI_CXX)

# ------------------------------------------------------------ installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS hyperforest EXPORT hyperforestTargets)
install(EXPORT hyperforestTargets NAMESPACE hyperforest::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hyperforest)
configure_package_config_file(cmake/hyperforestConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/hyperforestConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hyperforest)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/hyperforestConfigVersion.cmake
                                 COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hyperforestConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/hyperforestConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hyperforest)

# ------------------------------------------------------------ tests
if(NOT HYPERFOREST_ENABLE_TESTS)
  return()
endif()
enable_testing()
set(MPIRUN ${MPIEXEC_EXECUTABLE})

# core: brute-force checks of adapt / balance / partition / ghosts / face
# neighbours, D = 2 ... 5
add_executable(test_forest tests/test_forest.cc)
target_link_libraries(test_forest hyperforest)
foreach(np 1 3 4)
  add_test(NAME forest_np${np} COMMAND ${MPIRUN} -np ${np} $<TARGET_FILE:test_forest>)
endforeach()

if(NOT HYPERFOREST_WITH_DEALII)
  return()
endif()
find_package(deal.II 9.5.0 QUIET HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{HOME}/opt/dealii-9.5.2)
if(NOT deal.II_FOUND)
  message(STATUS "hyperforest: deal.II not found, skipping the deal.II adapter tests and examples")
  return()
endif()

# deal.II adapter: tree connectivity from deal.II meshes (rotated faces),
# phase space, partitioner, tensor-product DG basis
add_executable(test_dealii tests/test_dealii.cc)
target_link_libraries(test_dealii hyperforest)
target_compile_definitions(test_dealii PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
deal_ii_setup_target(test_dealii)
foreach(np 1 4)
  add_test(NAME dealii_np${np} COMMAND ${MPIRUN} -np ${np} $<TARGET_FILE:test_dealii>)
endforeach()

# example: free streaming in 2D / 4D / 6D phase space
add_executable(free_streaming examples/free_streaming.cc)
target_link_libraries(free_streaming hyperforest)
deal_ii_setup_target(free_streaming)
# the linear solution lies in the discrete space: exact to round-off on
# random nonconforming meshes (trees with rotated neighbours: mesh=ball)
foreach(spec "1;cube;1;3" "2;cube;1;3" "2;ball;1;4" "2;cube;2;3" "3;cube;1;4")
  list(GET spec 0 dim)
  list(GET spec 1 mesh)
  list(GET spec 2 degree)
  list(GET spec 3 np)
  set(cycles 3)
  if(dim EQUAL 3)
    set(cycles 2) # 6D: 5678 elements, 363k dofs in the second cycle
  endif()
  add_test(NAME free_streaming_linear_${dim}d_${mesh}_p${degree}_np${np}
           COMMAND ${MPIRUN} -np ${np} $<TARGET_FILE:free_streaming> dim=${dim} mesh=${mesh} degree=${degree}
                   test=linear refinement=random cycles=${cycles} tol=1e-15)
  set_tests_properties(free_streaming_linear_${dim}d_${mesh}_p${degree}_np${np} PROPERTIES
                       FAIL_REGULAR_EXPRESSION "e-0[0-9] *$|e-10 *$|Exception")
endforeach()
# smooth solution, uniform refinement: optimal L2 order p + 1 (1D x 1D)
foreach(spec "1;6;1\\.9[5-9]" "2;5;2\\.9[5-9]")
  list(GET spec 0 degree)
  list(GET spec 1 cycles)
  list(GET spec 2 rate)
  add_test(NAME free_streaming_order_p${degree} COMMAND ${MPIRUN} -np 2 $<TARGET_FILE:free_streaming> dim=1
           degree=${degree} test=smooth refinement=uniform cycles=${cycles})
  set_tests_properties(free_streaming_order_p${degree} PROPERTIES PASS_REGULAR_EXPRESSION "${rate} *\n[^\n]*$|${rate} *$")
endforeach()
