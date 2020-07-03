cmake_minimum_required(VERSION 3.13)
project(feature_extraction)

set(CMAKE_CXX_STANDARD 11)

#cuda
set(CUDA_ARCH sm_75)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_NVCC_FLAGS 
        "-arch=${CUDA_ARCH};--maxrregcount=30;-lineinfo;-std=c++11;-O2;"
        CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
find_package(CUDA REQUIRED)

#opencv
find_package(OpenCV REQUIRED)

#tensorrt
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

#link_directories(${OpenCV_LIBRARIES_DIRS}
 #                ${TENSORRT_LIBRARY_DIRS})
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        ${JSON_LIBRARY_DIRS}
	#${CUDA_LIBRARY_DIRS}
	${OpenCV_LIBRARIES_DIRS}
	${TENSORRT_INFER}
        ${TENSORRT_INFER_PLUGIN}
        )


include_directories(include
        ../common
        ${OpenCV_INCLUDE_DIRS}
	#${CUDA_INCLUDE_DIRS}
	#${CUDNN_INCLUDE_DIR}
	${TENSORRT_INCLUDE_DIR})

file(GLOB CPP_SRC src/*.cpp)
file(GLOB CU_SRC src/*.cu)

cuda_add_library(carFeature SHARED ${CPP_SRC} ${CU_SRC})
target_link_libraries(carFeature
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_highgui
        nvonnxparser
        nvonnxparser_runtime
        nvinfer
        nvinfer_plugin
        nvparsers
        #cudart
        )


cuda_add_executable(testCarFeature ut/testMain.cpp)
target_link_libraries(testCarFeature carFeature)

install(TARGETS testCarFeature carFeature
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
install(FILES include/CarFeature.h DESTINATION include)
