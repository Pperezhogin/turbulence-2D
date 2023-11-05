project(ProjectName)
cmake_minimum_required(VERSION 2.8)

enable_testing()

#set default compiler
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER mpicxx)

# define compiler and linker options
set (CMAKE_CXX_FLAGS "-qopenmp")

message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
# force Release for standard build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# debug options
# -check=uninit
set(CMAKE_CXX_FLAGS_DEBUG " -O0 -g -fstack-security-check -traceback -ftrapuv -debug all")

# release options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ")#-xcommon-avx512 -restrict -alias-const -funroll-loops -no-ansi-alias")

# headers folder
include_directories("${PROJECT_SOURCE_DIR}/include")

# FFTW library folders
#include_directories(${PROJECT_SOURCE_DIR}/fftw_library/include)
#link_directories(${PROJECT_SOURCE_DIR}/fftw_library/lib)

add_subdirectory(src)
add_subdirectory(test)
