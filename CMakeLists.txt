cmake_minimum_required(VERSION 3.2)

project(llm_simple CXX)
cmake_policy(SET CMP0128 OLD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

#add_subdirectory(src/npu_ops)
add_subdirectory(src)
