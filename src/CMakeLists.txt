
find_package (Eigen3 REQUIRED NO_MODULE)

file(GLOB SOURCE_FILES *.cpp)

add_executable(llama2_main ${SOURCE_FILES})
message("${EIGEN3_INCLUDE_DIR}")
target_include_directories(llama2_main PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(llama2_main sentencepiece boost_program_options boost_system boost_filesystem)
