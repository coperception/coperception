cmake_minimum_required(VERSION 3.4)
project(Pymodule)
SET(CMAKE_BUILD_TYPE "MinSizeRel")
set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 3.2.7 QUIET CONFIG)
find_package(pybind11 REQUIRED)

set(targets mapping)
foreach(target ${targets})
  pybind11_add_module(${target} THIN_LTO ${target}.cpp)
  target_link_libraries(${target} PRIVATE Eigen3::Eigen)
  if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                               "${CMAKE_CURRENT_BINARY_DIR}")

    if(DEFINED CMAKE_CONFIGURATION_TYPES)
      foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${config} config)
        set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${config}
                                                   "${CMAKE_CURRENT_BINARY_DIR}")
      endforeach()
    endif()
  endif()
endforeach()
target_link_libraries(mapping PRIVATE Eigen3::Eigen)