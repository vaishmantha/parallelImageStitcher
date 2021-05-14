cmake_minimum_required(VERSION 2.8)

file(GLOB gpu_source_files "image_stitching.cu")

file(GLOB_RECURSE EXAMPLE_SRC "image_stitching.cpp")
add_executable(image_stitching_cuda ${EXAMPLE_SRC})

target_link_libraries(image_stitching_cuda
	ezsift
)

if(CUDA_FOUND)
    message(STATUS "CUDA found to link")
    CUDA_ADD_LIBRARY(image_stitching_cuda_gpu ${gpu_source_files})
    target_link_libraries(image_stitching_cuda image_stitching_cuda_gpu)
endif()

