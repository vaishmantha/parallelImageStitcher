# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.18.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.18.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/vmantha/Desktop/parallelImageStitcher

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/vmantha/Desktop/parallelImageStitcher/build

# Include any dependencies generated for this target.
include bin/examples/image_match/CMakeFiles/image_match.dir/depend.make

# Include the progress variables for this target.
include bin/examples/image_match/CMakeFiles/image_match.dir/progress.make

# Include the compile flags for this target's objects.
include bin/examples/image_match/CMakeFiles/image_match.dir/flags.make

bin/examples/image_match/CMakeFiles/image_match.dir/image_match.cpp.o: bin/examples/image_match/CMakeFiles/image_match.dir/flags.make
bin/examples/image_match/CMakeFiles/image_match.dir/image_match.cpp.o: ../examples/image_match/image_match.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/vmantha/Desktop/parallelImageStitcher/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bin/examples/image_match/CMakeFiles/image_match.dir/image_match.cpp.o"
	cd /Users/vmantha/Desktop/parallelImageStitcher/build/bin/examples/image_match && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/image_match.dir/image_match.cpp.o -c /Users/vmantha/Desktop/parallelImageStitcher/examples/image_match/image_match.cpp

bin/examples/image_match/CMakeFiles/image_match.dir/image_match.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_match.dir/image_match.cpp.i"
	cd /Users/vmantha/Desktop/parallelImageStitcher/build/bin/examples/image_match && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vmantha/Desktop/parallelImageStitcher/examples/image_match/image_match.cpp > CMakeFiles/image_match.dir/image_match.cpp.i

bin/examples/image_match/CMakeFiles/image_match.dir/image_match.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_match.dir/image_match.cpp.s"
	cd /Users/vmantha/Desktop/parallelImageStitcher/build/bin/examples/image_match && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vmantha/Desktop/parallelImageStitcher/examples/image_match/image_match.cpp -o CMakeFiles/image_match.dir/image_match.cpp.s

# Object files for target image_match
image_match_OBJECTS = \
"CMakeFiles/image_match.dir/image_match.cpp.o"

# External object files for target image_match
image_match_EXTERNAL_OBJECTS =

bin/image_match: bin/examples/image_match/CMakeFiles/image_match.dir/image_match.cpp.o
bin/image_match: bin/examples/image_match/CMakeFiles/image_match.dir/build.make
bin/image_match: lib/libezsift.a
bin/image_match: bin/examples/image_match/CMakeFiles/image_match.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/vmantha/Desktop/parallelImageStitcher/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../image_match"
	cd /Users/vmantha/Desktop/parallelImageStitcher/build/bin/examples/image_match && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/image_match.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bin/examples/image_match/CMakeFiles/image_match.dir/build: bin/image_match

.PHONY : bin/examples/image_match/CMakeFiles/image_match.dir/build

bin/examples/image_match/CMakeFiles/image_match.dir/clean:
	cd /Users/vmantha/Desktop/parallelImageStitcher/build/bin/examples/image_match && $(CMAKE_COMMAND) -P CMakeFiles/image_match.dir/cmake_clean.cmake
.PHONY : bin/examples/image_match/CMakeFiles/image_match.dir/clean

bin/examples/image_match/CMakeFiles/image_match.dir/depend:
	cd /Users/vmantha/Desktop/parallelImageStitcher/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/vmantha/Desktop/parallelImageStitcher /Users/vmantha/Desktop/parallelImageStitcher/examples/image_match /Users/vmantha/Desktop/parallelImageStitcher/build /Users/vmantha/Desktop/parallelImageStitcher/build/bin/examples/image_match /Users/vmantha/Desktop/parallelImageStitcher/build/bin/examples/image_match/CMakeFiles/image_match.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bin/examples/image_match/CMakeFiles/image_match.dir/depend

