cmake_minimum_required(VERSION 3.0)

project(tinytrt)

set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE release)

set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive -fpic")

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")
option(BUILD_PYTHON "compile python api" OFF)

find_package(CUDA REQUIRED)
# Discover what architectures does nvcc support
include(cmake/CUDA_utils.cmake)
CUDA_find_supported_arch_values(CUDA_supported_archs ${CUDA_known_archs})
message(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
list(SORT CUDA_TARGET_ARCHS_SORTED)
CUDA_find_supported_arch_values(CUDA_targeted_archs ${CUDA_TARGET_ARCHS_SORTED})
message(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
if (NOT CUDA_targeted_archs)
  message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}}) is supported by nvcc, use one or more of: ${CUDA_supported_archs}")
endif()

CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
message(STATUS "Generated gencode flags: ${CUDA_gencode_flags}")

# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")

include_directories(spdlog)
include_directories(./)
#include_directories(../../../../thirdparty/TensorRT-7.1.3.4/include
#						${TENSORRT_INCLUDE_DIR})

set(TENSORRT_ROOT /home/TensorRT-6.0.1.5/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
find_path(TENSORRT_INFER libnvinfer.so
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_path(TENSORRT_INFER_PLUGIN libnvinfer_plugin.so
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
						
include_directories(${TENSORRT_INCLUDE_DIR})						

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda")

link_directories(${TENSORRT_LIBRARY_DIRS}
                 ${CUDA_LIBRARY_DIRS}
                 ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                 ${CUDNN_LIBRARY_DIRS})


file(GLOB_RECURSE trt_source
     Trt.cpp
     Int8EntropyCalibrator.cpp
	 ${TENSORRT_INCLUDE_DIR}/*.h
	 ./*.h
	 ./*.hpp
	 ./basewrapper.cpp
	 ./tensorrtwrapper.cpp
     )
cuda_add_library(tinytrt SHARED ${trt_source})
target_compile_options(tinytrt PUBLIC -std=c++11 -Wall -fpic -Wfloat-conversion)
set_target_properties(tinytrt PROPERTIES POSITION_INDEPENDENT_CODE ON)


