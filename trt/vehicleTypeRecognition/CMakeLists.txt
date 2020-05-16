cmake_minimum_required(VERSION 3.5)
project(VehicleType)

set(CMAKE_CXX_STANDARD 11)

#thirdparty
set(OpenCV_INCLUDE_DIRS ../thirdparty/opencv/include)
set(OpenCV_LIBRARIES_DIRS ../thirdparty/opencv/lib)

link_directories(${OpenCV_LIBRARIES_DIRS}
                 ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        )

include_directories(include
        ../common
        ${OpenCV_INCLUDE_DIRS}
        )

add_library(VehicleType SHARED src/VehicleType.cpp)
target_link_libraries(VehicleType
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_highgui
        )

install(TARGETS VehicleType
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
install(FILES include/VehicleType.h DESTINATION include)
