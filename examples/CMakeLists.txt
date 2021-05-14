cmake_minimum_required(VERSION 2.8)


add_subdirectory(image_stitching_seq)
add_subdirectory(image_stitching_cuda)

# file(GLOB_RECURSE TEST_IMAGES "../data/*.jpg")
# file(COPY ${TEST_IMAGES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(GLOB_RECURSE TEST_IMAGES "../data/*.png")
file(COPY ${TEST_IMAGES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

