cmake_minimum_required(VERSION 2.8)

file(GLOB_RECURSE EXAMPLE_SRC "image_stitching.cpp")
add_executable(image_stitching_pure_openmp ${EXAMPLE_SRC})

target_link_libraries(image_stitching_pure_openmp
	ezsift
)

