# CMake generated Testfile for 
# Source directory: /Users/vmantha/Desktop/parallelImageStitcher/include/opencv/modules/flann
# Build directory: /Users/vmantha/Desktop/parallelImageStitcher/include/build_opencv/modules/flann
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_flann "/Users/vmantha/Desktop/parallelImageStitcher/include/build_opencv/bin/opencv_test_flann" "--gtest_output=xml:opencv_test_flann.xml")
set_tests_properties(opencv_test_flann PROPERTIES  LABELS "Main;opencv_flann;Accuracy" WORKING_DIRECTORY "/Users/vmantha/Desktop/parallelImageStitcher/include/build_opencv/test-reports/accuracy" _BACKTRACE_TRIPLES "/Users/vmantha/Desktop/parallelImageStitcher/include/opencv/cmake/OpenCVUtils.cmake;1707;add_test;/Users/vmantha/Desktop/parallelImageStitcher/include/opencv/cmake/OpenCVModule.cmake;1315;ocv_add_test_from_target;/Users/vmantha/Desktop/parallelImageStitcher/include/opencv/cmake/OpenCVModule.cmake;1079;ocv_add_accuracy_tests;/Users/vmantha/Desktop/parallelImageStitcher/include/opencv/modules/flann/CMakeLists.txt;2;ocv_define_module;/Users/vmantha/Desktop/parallelImageStitcher/include/opencv/modules/flann/CMakeLists.txt;0;")
