cmake_minimum_required(VERSION 2.8)

# option(USE_CUDA "Use CUDA" OFF) #SWITCH TO ON WHEN RUNNING ON GPU
file(GLOB gpu_source_files "image_stitching.cu")

file(GLOB_RECURSE EXAMPLE_SRC "image_stitching.cpp")
add_executable(image_stitching_cuda ${EXAMPLE_SRC})

target_link_libraries(image_stitching_cuda
	ezsift
)

# if(USE_CUDA)
#     find_package(CUDA 10.2)
#     if(NOT CUDA_FOUND)
#         message(STATUS "CUDA not found")
#         set(USE_CUDA OFF)
#     else()
#         CUDA_ADD_LIBRARY(image_stitching_cuda_gpu ${gpu_source_files})
#         target_link_libraries(image_stitching_cuda image_stitching_cuda_gpu)
#     endif()
# endif()
if(CUDA_FOUND)
    CUDA_ADD_LIBRARY(image_stitching_cuda_gpu ${gpu_source_files})
    target_link_libraries(image_stitching_cuda image_stitching_cuda_gpu)
endif()

# if(USE_CUDA)
#     add_definitions(-DUSE_CUDA)
# endif()
