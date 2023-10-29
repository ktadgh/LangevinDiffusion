cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(LangevinDiffusion)

set(CMAKE_PREFIX_PATH "/home/tadgh/libtorch")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



add_executable(main main.cpp)

# Set compiler options for the 'main' target
target_compile_options(main PRIVATE
    -g
    #-Ofast #optimizes
    -funroll-loops
    -fopt-info-vec-missed
                  # optimizes
    # Add more options as needed
)

target_include_directories(main PRIVATE ${TORCH_INCLUDE_DIRS})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(main ${PYTHON_LIBRARIES})


set_property(TARGET main PROPERTY CXX_STANDARD 17)
