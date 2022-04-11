cmake_minimum_required(VERSION 3.4)

project(roll)

set(TENSORRT_DIR "D:\\TensorRT-7.2.2.3")

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")
if(WIN32)
include_directories(${PROJECT_SOURCE_DIR}/include)
endif(WIN32)


find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode)

# cuda
enable_language(CUDA)  

# tensorrt
include_directories(${TENSORRT_DIR}\\include)
link_directories(${TENSORRT_DIR}\\lib)



file(GLOB SOURCE_FILES "*.h" "*.cpp" "*.cu")
add_executable(roll  ${SOURCE_FILES})

target_link_libraries(roll nvinfer nvinfer_plugin)
target_link_libraries(roll cudart cuda)


if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)