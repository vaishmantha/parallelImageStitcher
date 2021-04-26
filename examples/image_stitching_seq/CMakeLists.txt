cmake_minimum_required(VERSION 2.8)

file(GLOB_RECURSE EXAMPLE_SRC "image_stitching.cpp")
add_executable(image_stitching_seq ${EXAMPLE_SRC})

target_link_libraries(image_stitching_seq
	ezsift
)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries(image_stitching_seq
	${OpenCV_LIBS}
)
