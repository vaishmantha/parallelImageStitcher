# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.19.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.19.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/adithirao/Documents/parallel/ezSIFT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/adithirao/Documents/parallel/ezSIFT/platforms

# Include any dependencies generated for this target.
include CMakeFiles/ezsift.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ezsift.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ezsift.dir/flags.make

CMakeFiles/ezsift.dir/src/ezsift.cpp.o: CMakeFiles/ezsift.dir/flags.make
CMakeFiles/ezsift.dir/src/ezsift.cpp.o: ../src/ezsift.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/adithirao/Documents/parallel/ezSIFT/platforms/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ezsift.dir/src/ezsift.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ezsift.dir/src/ezsift.cpp.o -c /Users/adithirao/Documents/parallel/ezSIFT/src/ezsift.cpp

CMakeFiles/ezsift.dir/src/ezsift.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ezsift.dir/src/ezsift.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/adithirao/Documents/parallel/ezSIFT/src/ezsift.cpp > CMakeFiles/ezsift.dir/src/ezsift.cpp.i

CMakeFiles/ezsift.dir/src/ezsift.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ezsift.dir/src/ezsift.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/adithirao/Documents/parallel/ezSIFT/src/ezsift.cpp -o CMakeFiles/ezsift.dir/src/ezsift.cpp.s

CMakeFiles/ezsift.dir/src/image_utility.cpp.o: CMakeFiles/ezsift.dir/flags.make
CMakeFiles/ezsift.dir/src/image_utility.cpp.o: ../src/image_utility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/adithirao/Documents/parallel/ezSIFT/platforms/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ezsift.dir/src/image_utility.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ezsift.dir/src/image_utility.cpp.o -c /Users/adithirao/Documents/parallel/ezSIFT/src/image_utility.cpp

CMakeFiles/ezsift.dir/src/image_utility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ezsift.dir/src/image_utility.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/adithirao/Documents/parallel/ezSIFT/src/image_utility.cpp > CMakeFiles/ezsift.dir/src/image_utility.cpp.i

CMakeFiles/ezsift.dir/src/image_utility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ezsift.dir/src/image_utility.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/adithirao/Documents/parallel/ezSIFT/src/image_utility.cpp -o CMakeFiles/ezsift.dir/src/image_utility.cpp.s

CMakeFiles/ezsift.dir/src/img-stitch.cpp.o: CMakeFiles/ezsift.dir/flags.make
CMakeFiles/ezsift.dir/src/img-stitch.cpp.o: ../src/img-stitch.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/adithirao/Documents/parallel/ezSIFT/platforms/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ezsift.dir/src/img-stitch.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ezsift.dir/src/img-stitch.cpp.o -c /Users/adithirao/Documents/parallel/ezSIFT/src/img-stitch.cpp

CMakeFiles/ezsift.dir/src/img-stitch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ezsift.dir/src/img-stitch.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/adithirao/Documents/parallel/ezSIFT/src/img-stitch.cpp > CMakeFiles/ezsift.dir/src/img-stitch.cpp.i

CMakeFiles/ezsift.dir/src/img-stitch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ezsift.dir/src/img-stitch.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/adithirao/Documents/parallel/ezSIFT/src/img-stitch.cpp -o CMakeFiles/ezsift.dir/src/img-stitch.cpp.s

# Object files for target ezsift
ezsift_OBJECTS = \
"CMakeFiles/ezsift.dir/src/ezsift.cpp.o" \
"CMakeFiles/ezsift.dir/src/image_utility.cpp.o" \
"CMakeFiles/ezsift.dir/src/img-stitch.cpp.o"

# External object files for target ezsift
ezsift_EXTERNAL_OBJECTS =

lib/libezsift.a: CMakeFiles/ezsift.dir/src/ezsift.cpp.o
lib/libezsift.a: CMakeFiles/ezsift.dir/src/image_utility.cpp.o
lib/libezsift.a: CMakeFiles/ezsift.dir/src/img-stitch.cpp.o
lib/libezsift.a: CMakeFiles/ezsift.dir/build.make
lib/libezsift.a: CMakeFiles/ezsift.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/adithirao/Documents/parallel/ezSIFT/platforms/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library lib/libezsift.a"
	$(CMAKE_COMMAND) -P CMakeFiles/ezsift.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ezsift.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ezsift.dir/build: lib/libezsift.a

.PHONY : CMakeFiles/ezsift.dir/build

CMakeFiles/ezsift.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ezsift.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ezsift.dir/clean

CMakeFiles/ezsift.dir/depend:
	cd /Users/adithirao/Documents/parallel/ezSIFT/platforms && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/adithirao/Documents/parallel/ezSIFT /Users/adithirao/Documents/parallel/ezSIFT /Users/adithirao/Documents/parallel/ezSIFT/platforms /Users/adithirao/Documents/parallel/ezSIFT/platforms /Users/adithirao/Documents/parallel/ezSIFT/platforms/CMakeFiles/ezsift.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ezsift.dir/depend

