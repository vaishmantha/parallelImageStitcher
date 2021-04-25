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
CMAKE_BINARY_DIR = /Users/adithirao/Documents/parallel/ezSIFT/build

# Include any dependencies generated for this target.
include bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/depend.make

# Include the progress variables for this target.
include bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/progress.make

# Include the compile flags for this target's objects.
include bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/flags.make

bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.o: bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/flags.make
bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.o: ../examples/image_stitching_seq/image_stitching.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/adithirao/Documents/parallel/ezSIFT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.o"
	cd /Users/adithirao/Documents/parallel/ezSIFT/build/bin/examples/image_stitching_seq && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.o -c /Users/adithirao/Documents/parallel/ezSIFT/examples/image_stitching_seq/image_stitching.cpp

bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.i"
	cd /Users/adithirao/Documents/parallel/ezSIFT/build/bin/examples/image_stitching_seq && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/adithirao/Documents/parallel/ezSIFT/examples/image_stitching_seq/image_stitching.cpp > CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.i

bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.s"
	cd /Users/adithirao/Documents/parallel/ezSIFT/build/bin/examples/image_stitching_seq && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/adithirao/Documents/parallel/ezSIFT/examples/image_stitching_seq/image_stitching.cpp -o CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.s

# Object files for target image_stitching_seq
image_stitching_seq_OBJECTS = \
"CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.o"

# External object files for target image_stitching_seq
image_stitching_seq_EXTERNAL_OBJECTS =

bin/image_stitching_seq: bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/image_stitching.cpp.o
bin/image_stitching_seq: bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/build.make
bin/image_stitching_seq: lib/libezsift.a
bin/image_stitching_seq: bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/adithirao/Documents/parallel/ezSIFT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../image_stitching_seq"
	cd /Users/adithirao/Documents/parallel/ezSIFT/build/bin/examples/image_stitching_seq && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/image_stitching_seq.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/build: bin/image_stitching_seq

.PHONY : bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/build

bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/clean:
	cd /Users/adithirao/Documents/parallel/ezSIFT/build/bin/examples/image_stitching_seq && $(CMAKE_COMMAND) -P CMakeFiles/image_stitching_seq.dir/cmake_clean.cmake
.PHONY : bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/clean

bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/depend:
	cd /Users/adithirao/Documents/parallel/ezSIFT/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/adithirao/Documents/parallel/ezSIFT /Users/adithirao/Documents/parallel/ezSIFT/examples/image_stitching_seq /Users/adithirao/Documents/parallel/ezSIFT/build /Users/adithirao/Documents/parallel/ezSIFT/build/bin/examples/image_stitching_seq /Users/adithirao/Documents/parallel/ezSIFT/build/bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bin/examples/image_stitching_seq/CMakeFiles/image_stitching_seq.dir/depend

