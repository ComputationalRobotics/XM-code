# example of CUOPT
cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

project(XM LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# # Global CXX flags/options
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# # Global CUDA CXX flags/options
# set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
# set(CMAKE_CUDA_STANDARD 11)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_EXTENSIONS OFF)

# Debug options

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -lineinfo")
add_compile_options(-Wno-unknown-pragmas)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# set(Python_FIND_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Found Python3 interpreter: ${Python3_EXECUTABLE}")
message(STATUS "Python3 version: ${Python3_VERSION}")
message(STATUS "Python3 include directories: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")

include_directories(${Python3_INCLUDE_DIRS})
link_libraries(${Python3_LIBRARIES})

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)

FetchContent_MakeAvailable(pybind11)

include("${CMAKE_SOURCE_DIR}/cmake/CUOPT_example.cmake")
include_directories("${CMAKE_SOURCE_DIR}/include")

add_example(XM "XM" src/XM_main.cu)

