# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/arrouan/clion/clion-2018.1.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/arrouan/clion/clion-2018.1.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/arrouan/workspace/micro_aevol/mini-aevol

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/arrouan/workspace/micro_aevol/mini-aevol

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/home/arrouan/clion/clion-2018.1.3/bin/cmake/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/home/arrouan/clion/clion-2018.1.3/bin/cmake/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/arrouan/workspace/micro_aevol/mini-aevol/CMakeFiles /home/arrouan/workspace/micro_aevol/mini-aevol/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/arrouan/workspace/micro_aevol/mini-aevol/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named pdc_mini_aevol

# Build rule for target.
pdc_mini_aevol: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 pdc_mini_aevol
.PHONY : pdc_mini_aevol

# fast build rule for target.
pdc_mini_aevol/fast:
	$(MAKE) -f CMakeFiles/pdc_mini_aevol.dir/build.make CMakeFiles/pdc_mini_aevol.dir/build
.PHONY : pdc_mini_aevol/fast

#=============================================================================
# Target rules for targets named mini_aevol_gpu

# Build rule for target.
mini_aevol_gpu: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 mini_aevol_gpu
.PHONY : mini_aevol_gpu

# fast build rule for target.
mini_aevol_gpu/fast:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/build
.PHONY : mini_aevol_gpu/fast

#=============================================================================
# Target rules for targets named sfmt

# Build rule for target.
sfmt: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 sfmt
.PHONY : sfmt

# fast build rule for target.
sfmt/fast:
	$(MAKE) -f SFMT-src-1.4/CMakeFiles/sfmt.dir/build.make SFMT-src-1.4/CMakeFiles/sfmt.dir/build
.PHONY : sfmt/fast

AeTime.o: AeTime.cpp.o

.PHONY : AeTime.o

# target to build an object file
AeTime.cpp.o:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/AeTime.cpp.o
.PHONY : AeTime.cpp.o

AeTime.i: AeTime.cpp.i

.PHONY : AeTime.i

# target to preprocess a source file
AeTime.cpp.i:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/AeTime.cpp.i
.PHONY : AeTime.cpp.i

AeTime.s: AeTime.cpp.s

.PHONY : AeTime.s

# target to generate assembly for a file
AeTime.cpp.s:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/AeTime.cpp.s
.PHONY : AeTime.cpp.s

Dna.o: Dna.cpp.o

.PHONY : Dna.o

# target to build an object file
Dna.cpp.o:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Dna.cpp.o
.PHONY : Dna.cpp.o

Dna.i: Dna.cpp.i

.PHONY : Dna.i

# target to preprocess a source file
Dna.cpp.i:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Dna.cpp.i
.PHONY : Dna.cpp.i

Dna.s: Dna.cpp.s

.PHONY : Dna.s

# target to generate assembly for a file
Dna.cpp.s:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Dna.cpp.s
.PHONY : Dna.cpp.s

DnaMutator.o: DnaMutator.cpp.o

.PHONY : DnaMutator.o

# target to build an object file
DnaMutator.cpp.o:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/DnaMutator.cpp.o
.PHONY : DnaMutator.cpp.o

DnaMutator.i: DnaMutator.cpp.i

.PHONY : DnaMutator.i

# target to preprocess a source file
DnaMutator.cpp.i:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/DnaMutator.cpp.i
.PHONY : DnaMutator.cpp.i

DnaMutator.s: DnaMutator.cpp.s

.PHONY : DnaMutator.s

# target to generate assembly for a file
DnaMutator.cpp.s:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/DnaMutator.cpp.s
.PHONY : DnaMutator.cpp.s

ExpManager.o: ExpManager.cpp.o

.PHONY : ExpManager.o

# target to build an object file
ExpManager.cpp.o:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/ExpManager.cpp.o
.PHONY : ExpManager.cpp.o

ExpManager.i: ExpManager.cpp.i

.PHONY : ExpManager.i

# target to preprocess a source file
ExpManager.cpp.i:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/ExpManager.cpp.i
.PHONY : ExpManager.cpp.i

ExpManager.s: ExpManager.cpp.s

.PHONY : ExpManager.s

# target to generate assembly for a file
ExpManager.cpp.s:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/ExpManager.cpp.s
.PHONY : ExpManager.cpp.s

JumpingMT.o: JumpingMT.cpp.o

.PHONY : JumpingMT.o

# target to build an object file
JumpingMT.cpp.o:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/JumpingMT.cpp.o
.PHONY : JumpingMT.cpp.o

JumpingMT.i: JumpingMT.cpp.i

.PHONY : JumpingMT.i

# target to preprocess a source file
JumpingMT.cpp.i:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/JumpingMT.cpp.i
.PHONY : JumpingMT.cpp.i

JumpingMT.s: JumpingMT.cpp.s

.PHONY : JumpingMT.s

# target to generate assembly for a file
JumpingMT.cpp.s:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/JumpingMT.cpp.s
.PHONY : JumpingMT.cpp.s

MutationEvent.o: MutationEvent.cpp.o

.PHONY : MutationEvent.o

# target to build an object file
MutationEvent.cpp.o:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/MutationEvent.cpp.o
.PHONY : MutationEvent.cpp.o

MutationEvent.i: MutationEvent.cpp.i

.PHONY : MutationEvent.i

# target to preprocess a source file
MutationEvent.cpp.i:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/MutationEvent.cpp.i
.PHONY : MutationEvent.cpp.i

MutationEvent.s: MutationEvent.cpp.s

.PHONY : MutationEvent.s

# target to generate assembly for a file
MutationEvent.cpp.s:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/MutationEvent.cpp.s
.PHONY : MutationEvent.cpp.s

Organism.o: Organism.cpp.o

.PHONY : Organism.o

# target to build an object file
Organism.cpp.o:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Organism.cpp.o
.PHONY : Organism.cpp.o

Organism.i: Organism.cpp.i

.PHONY : Organism.i

# target to preprocess a source file
Organism.cpp.i:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Organism.cpp.i
.PHONY : Organism.cpp.i

Organism.s: Organism.cpp.s

.PHONY : Organism.s

# target to generate assembly for a file
Organism.cpp.s:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Organism.cpp.s
.PHONY : Organism.cpp.s

Stats.o: Stats.cpp.o

.PHONY : Stats.o

# target to build an object file
Stats.cpp.o:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Stats.cpp.o
.PHONY : Stats.cpp.o

Stats.i: Stats.cpp.i

.PHONY : Stats.i

# target to preprocess a source file
Stats.cpp.i:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Stats.cpp.i
.PHONY : Stats.cpp.i

Stats.s: Stats.cpp.s

.PHONY : Stats.s

# target to generate assembly for a file
Stats.cpp.s:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Stats.cpp.s
.PHONY : Stats.cpp.s

Threefry.o: Threefry.cpp.o

.PHONY : Threefry.o

# target to build an object file
Threefry.cpp.o:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Threefry.cpp.o
.PHONY : Threefry.cpp.o

Threefry.i: Threefry.cpp.i

.PHONY : Threefry.i

# target to preprocess a source file
Threefry.cpp.i:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Threefry.cpp.i
.PHONY : Threefry.cpp.i

Threefry.s: Threefry.cpp.s

.PHONY : Threefry.s

# target to generate assembly for a file
Threefry.cpp.s:
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/Threefry.cpp.s
.PHONY : Threefry.cpp.s

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/pdc_mini_aevol.dir/build.make CMakeFiles/pdc_mini_aevol.dir/main.cpp.o
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/pdc_mini_aevol.dir/build.make CMakeFiles/pdc_mini_aevol.dir/main.cpp.i
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/pdc_mini_aevol.dir/build.make CMakeFiles/pdc_mini_aevol.dir/main.cpp.s
	$(MAKE) -f CMakeFiles/mini_aevol_gpu.dir/build.make CMakeFiles/mini_aevol_gpu.dir/main.cpp.s
.PHONY : main.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... pdc_mini_aevol"
	@echo "... rebuild_cache"
	@echo "... mini_aevol_gpu"
	@echo "... sfmt"
	@echo "... AeTime.o"
	@echo "... AeTime.i"
	@echo "... AeTime.s"
	@echo "... Dna.o"
	@echo "... Dna.i"
	@echo "... Dna.s"
	@echo "... DnaMutator.o"
	@echo "... DnaMutator.i"
	@echo "... DnaMutator.s"
	@echo "... ExpManager.o"
	@echo "... ExpManager.i"
	@echo "... ExpManager.s"
	@echo "... JumpingMT.o"
	@echo "... JumpingMT.i"
	@echo "... JumpingMT.s"
	@echo "... MutationEvent.o"
	@echo "... MutationEvent.i"
	@echo "... MutationEvent.s"
	@echo "... Organism.o"
	@echo "... Organism.i"
	@echo "... Organism.s"
	@echo "... Stats.o"
	@echo "... Stats.i"
	@echo "... Stats.s"
	@echo "... Threefry.o"
	@echo "... Threefry.i"
	@echo "... Threefry.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
