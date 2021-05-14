cmake_minimum_required(VERSION 2.8)
set(CMAKE_C_COMPILER "/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++") #local version
#set(CMAKE_CXX_COMPILER "/opt/rh/devtoolset-9/root/bin/g++") #ghc version
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(EZSIFT ezsift)
project(${EZSIFT})

option(USE_CUDA "Use CUDA" ON) #SWITCH TO ON WHEN RUNNING ON GPU
option(EZSIFT_BUILD_EXAMPLES "Build with examples" ON)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endforeach()

set(EZSIFT_ROOT ${PROJECT_SOURCE_DIR})
set(EZSIFT_INCLUDE_DIR ${EZSIFT_ROOT}/include)
set(EZSIFT_SRC_DIR ${EZSIFT_ROOT}/src)

file(GLOB_RECURSE SRC RELATIVE ${PROJECT_SOURCE_DIR} "${EZSIFT_SRC_DIR}/*.cpp")
file(GLOB_RECURSE HEADERS RELATIVE ${PROJECT_SOURCE_DIR} "${EZSIFT_SRC_DIR}/*.h" "${EZSIFT_INCLUDE_DIR}/*.h")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(${EZSIFT} ${SRC})

add_library(lodepng include/lodepng/lodepng.cc)
target_link_libraries(${EZSIFT} ${JPEG_LIBRARY} lodepng)
set_property(TARGET ${EZSIFT} PROPERTY POSITION_INDEPENDENT_CODE ON)

include_directories(
    ${EZSIFT_INCLUDE_DIR}
)

target_include_directories(${EZSIFT} PUBLIC ${EZSIFT_SRC_DIR})

if(EZSIFT_BUILD_EXAMPLES)
  add_subdirectory(${EZSIFT_ROOT}/examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples/)
endif()

if(USE_CUDA)
    find_package(CUDA 10.2)
    if(NOT CUDA_FOUND)
        message(STATUS "CUDA not found")
        set(USE_CUDA OFF)
    endif()
endif()

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
