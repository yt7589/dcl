cmake_minimum_required(VERSION 3.5)
project(fgvc_predictor_proj)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE release)

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -fpic")


set(GPU_ARCHS 61)  ## config your GPU_ARCHS
#set(TENSORRT_ROOT /home/zhangsy/TensorRT-5.0.2.6)

		 
find_package(OpenCV REQUIRED)	

FIND_PACKAGE(Protobuf REQUIRED)

message(status " " ${PROTOBUF_LIBRARY})

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER}
         nvonnxparser nvonnxparser_runtime)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpic -Wall -Ofast -Wl,-Bsymbolic")
include_directories(./install/include
					${TENSORRT_INCLUDE_DIR}
					../src
					../../tensorrtcppwrapper
					.)
#add_subdirectory(src)
link_directories(
		/home/zhangsy/tensorrt7/TensorRT/parsers/onnx/build
		/home/zhangsy/tensorrt7/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib
		/home/zhangsy/tensorrt7/tiny-tensorrt/build
		../../lib
		)
		
file(GLOB CPP_SRC ./*.cpp)
file(GLOB CU_SRC ./*.cu)

add_library(fgvc_predictor  SHARED ${CPP_SRC})
target_link_libraries(fgvc_predictor  ${OpenCV_LIBS} tensorrtwrapper  tinytrt nvinfer  protobuf )
target_compile_options(fgvc_predictor  PUBLIC -std=c++11 -Wall -fpic  -Wfloat-conversion)

target_link_libraries(fgvc_predictor  nvinfer)
target_link_libraries(fgvc_predictor  nvinfer_plugin)
target_link_libraries(fgvc_predictor  nvparsers)
target_link_libraries(fgvc_predictor  nvonnxparser)
#target_link_libraries(fgvc_predictor  nvonnxparser_runtime)


target_link_libraries(fgvc_predictor nvcaffe_parser)
target_link_libraries(fgvc_predictor ${CUDART})
target_link_libraries(fgvc_predictor ${OpenCV_LIBRARIES})


