cmake_minimum_required(VERSION 3.21)
project(autoluce_nvfp4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_library(autoluce_nvfp4 STATIC nvfp4.cu)
target_include_directories(autoluce_nvfp4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(autoluce_nvfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(test_nvfp4 test_nvfp4.cu)
target_link_libraries(test_nvfp4 PRIVATE autoluce_nvfp4)

add_executable(bench_nvfp4 bench_nvfp4.cu)
target_link_libraries(bench_nvfp4 PRIVATE autoluce_nvfp4)

enable_testing()
add_test(NAME nvfp4_cuda_oracle COMMAND test_nvfp4)
