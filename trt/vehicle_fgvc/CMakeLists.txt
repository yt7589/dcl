cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME vehicle_fgvc)
project(${PROJECT_NAME})
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)

#cuda
option(ENABLE_CUDA "" ON)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

set(CUDA_ARCH sm_61)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_NVCC_FLAGS
        "-arch=${CUDA_ARCH};--maxrregcount=30;-lineinfo"
        CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

find_package(CUDA REQUIRED)
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

set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)

set(CUDNN_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu/)


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${CUDNN_INCLUDE_DIR}
)

set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64/)
link_directories(${OpenCV_LIBRARIES_DIRS}
                 ${TENSORRT_LIBRARY_DIRS}
                 ${TENSORRT_ROOT}/lib
                 ${CUDA_LIBRARY_DIRS}
                 ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                 ${CUDNN_LIBRARY_DIRS})
include_directories(include
        ../decode/include
        ../common
		../encryption/include
        )


file(GLOB CPP_SRC_BASE src/tensorrtcppwrapper/*.cpp)
file(GLOB CU_SRC_BASE src/tensorrtcppwrapper/*.cu)

file(GLOB CPP_SRC src/*.cpp)
file(GLOB CU_SRC src/*.cu)

file(GLOB CPP_SRC_COMMON ../common/file_util.cpp)
cuda_add_library(tinytrt_v2 SHARED ${CPP_SRC_BASE} ${CU_SRC_BASE}
					${CPP_SRC_COMMON})

cuda_add_library(vehicle_fgvc_v1 SHARED ${CPP_SRC} ${CU_SRC})

target_link_libraries( vehicle_fgvc_v1 tinytrt_v2
		 opencv_core opencv_imgcodecs opencv_imgproc
		cudart)

target_link_libraries( tinytrt_v2
		 opencv_core opencv_imgcodecs opencv_imgproc
		cudart)     

target_link_libraries(tinytrt_v2 nvinfer)
target_link_libraries(tinytrt_v2  nvinfer_plugin)
target_link_libraries(tinytrt_v2  nvparsers)
target_link_libraries(tinytrt_v2  nvonnxparser)
#target_link_libraries(tinytrt_v2  nvonnxparser_runtime)
           

# nvjpeg nppig nppc culibos nvjpeg nppisu

#cuda_add_executable(test_vehicle_fgvc ut/main.cpp)
#target_link_libraries(test_vehicle_fgvc pthread vehicle_fgvc_v1 tinytrt_v2)

cuda_add_executable(cropMain ut/cropMain.cpp)
target_link_libraries(cropMain pthread vehicle_fgvc_v1 tinytrt_v2)

install(TARGETS vehicle_fgvc_v1 tinytrt_v2 cropMain
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
install(FILES include/vehicle_fgvc.h DESTINATION include)
