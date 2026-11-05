cmake_minimum_required(VERSION 3.16)

project(simple_rtmpose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT
    "/home/ubuntu20/tensorrt-local/TensorRT-8.6.1.6"
    CACHE PATH
    "TensorRT installation directory")
set(CUDA_ROOT
    "/usr/local/cuda-11.7"
    CACHE PATH
    "CUDA installation directory")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS "${TENSORRT_ROOT}/include")
find_library(NVINFER_LIBRARY nvinfer
    HINTS "${TENSORRT_ROOT}/lib")
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
    HINTS "${CUDA_ROOT}/include")
find_library(CUDART_LIBRARY cudart
    HINTS "${CUDA_ROOT}/lib64")

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIBRARY OR
    NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIBRARY)
    message(FATAL_ERROR "找不到 TensorRT 或 CUDA，請檢查 TENSORRT_ROOT/CUDA_ROOT")
endif()

add_library(rtmpose src/rtmpose.cpp)
target_include_directories(rtmpose
    PUBLIC include ${OpenCV_INCLUDE_DIRS}
    PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CUDA_INCLUDE_DIR}")
target_link_libraries(rtmpose
    PUBLIC ${OpenCV_LIBS}
    PRIVATE ${NVINFER_LIBRARY} ${CUDART_LIBRARY})
target_compile_options(rtmpose PRIVATE -Wno-deprecated-declarations)

add_executable(rtmpose_app
    src/main.cpp
    src/inputUtils.cpp)
target_link_libraries(rtmpose_app PRIVATE rtmpose)

set_target_properties(rtmpose_app PROPERTIES
    BUILD_RPATH "${TENSORRT_ROOT}/lib;${CUDA_ROOT}/lib64")

