cmake_minimum_required(VERSION 2.8)


add_subdirectory(image_stitching_seq)
add_subdirectory(image_stitching_pure_openmp)
if(CUDA_FOUND)
    add_subdirectory(image_stitching_cuda)
endif()


file(GLOB_RECURSE TEST_IMAGES "../data/*.png")
file(COPY ${TEST_IMAGES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

