cmake_minimum_required(VERSION 3.10)
project(cascade_runner)

set(CMAKE_CXX_STANDARD 17)

# Path to ONNX Runtime directory
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.17.3")

# Add includes and libs directly
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Add executable
add_executable(predictor predictor.cpp)

# Link ONNX Runtime shared library (libonnxruntime.so)
target_link_libraries(predictor PRIVATE onnxruntime)
