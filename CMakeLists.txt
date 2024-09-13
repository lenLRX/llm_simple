cmake_minimum_required(VERSION 3.2)

project(llm_simple CXX)
cmake_policy(SET CMP0128 OLD)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

add_subdirectory(src)
add_subdirectory(tests)