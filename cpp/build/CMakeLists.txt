cmake_minimum_required(VERSION 3.10)
project(cascade_runner)

set(CMAKE_CXX_STANDARD 17)

# Specify the ONNXRuntime directory
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.17.3")
set(CMAKE_PREFIX_PATH "${ONNXRUNTIME_DIR}/lib/cmake/onnxruntime")

# Find the ONNX Runtime package
find_package(ONNXRuntime REQUIRED)

# Add executable
add_executable(predictor predictor.cpp)

# Include ONNX Runtime headers
target_include_directories(predictor PRIVATE
    ${ONNXRUNTIME_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}  # Optional, provided by find_package
)

# Link ONNX Runtime library
target_link_libraries(predictor PRIVATE
    onnxruntime  # CMake target name usually provided by ONNXRuntime
)
