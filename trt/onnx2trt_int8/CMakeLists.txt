cmake_minimum_required(VERSION 3.5)
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

#thirdparty
set(OpenCV_INCLUDE_DIRS ../thirdparty/opencv-3.4.7/include)
set(OpenCV_LIBRARIES_DIRS ../thirdparty/opencv-3.4.7/build/lib)

set(TENSORRT_INCLUDE_DIRS ../thirdparty/TensorRT-6.0.1.5/include)
set(TENSORRT_LIBRARY_DIRS ../thirdparty/TensorRT-6.0.1.5/lib)

link_directories(${OpenCV_LIBRARIES_DIRS}
                 ${TENSORRT_LIBRARY_DIRS})

include_directories(include
        ../common
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CUDNN_INCLUDE_DIR}
	${TENSORRT_INCLUDE_DIRS})

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
