if (DEFINED ACL_PATH)
        message(STATUS "user set ACL_PATH: ${ACL_PATH}")
elseif (EXISTS /usr/local/Ascend/acllib/)
        set(ACL_PATH "/usr/local/Ascend/acllib")
        message(STATUS "set ACL_PATH: /usr/local/Ascend/acllib")
elseif (EXISTS /usr/local/Ascend/ascend-toolkit/latest/acllib)
        set(ACL_PATH "/usr/local/Ascend/ascend-toolkit/latest/acllib")
        message(STATUS "set ACL_PATH to default path: /usr/local/Ascend/ascend-toolkit/latest/acllib")
elseif (EXISTS /usr/local/Ascend/nnrt/latest/acllib)
        set(ACL_PATH "/usr/local/Ascend/nnrt/latest/acllib")
        message(STATUS "set ACL_PATH to default path: /usr/local/Ascend/nnrt/latest/acllib")    
else ()
        set(ACL_PATH "/home/HwHiAiUser/Ascend/acllib")
        message(STATUS "set ACL_PATH to default path: /home/HwHiAiUser/Ascend/acllib")    
endif()


find_package (Eigen3 REQUIRED NO_MODULE)

enable_testing()
find_package(GTest REQUIRED)

add_executable(op_test npu_operator_test.cpp rms_norm_layer_test.cpp gemm_awq_4bit_test.cpp npu_op_test_util.cpp)
target_include_directories(op_test PUBLIC ${EIGEN3_INCLUDE_DIR} ${ACL_PATH}/include ../prebuild)
target_link_directories(op_test PUBLIC ${ACL_PATH}/lib64 ../prebuild)
target_link_libraries(op_test GTest::gtest GTest::gtest_main sentencepiece boost_program_options boost_system boost_filesystem fmt ascendcl runtime npu_ops)

add_test(NPU_operator_test op_test)

add_executable(op_test_rms_norm rms_norm_layer_main.cpp rms_norm_layer_test.cpp npu_op_test_util.cpp)
target_include_directories(op_test_rms_norm PUBLIC ${EIGEN3_INCLUDE_DIR} ${ACL_PATH}/include ../prebuild)
target_link_directories(op_test_rms_norm PUBLIC ${ACL_PATH}/lib64 ../prebuild)
target_link_libraries(op_test_rms_norm  sentencepiece boost_program_options boost_system boost_filesystem fmt ascendcl runtime npu_ops)

add_executable(op_test_gemm_awq_4bit gemm_awq_4bit_main.cpp gemm_awq_4bit_test.cpp npu_op_test_util.cpp)
target_include_directories(op_test_gemm_awq_4bit PUBLIC ${EIGEN3_INCLUDE_DIR} ${ACL_PATH}/include ../prebuild)
target_link_directories(op_test_gemm_awq_4bit PUBLIC ${ACL_PATH}/lib64 ../prebuild)
target_link_libraries(op_test_gemm_awq_4bit  sentencepiece boost_program_options boost_system boost_filesystem fmt ascendcl runtime npu_ops)

add_executable(op_test_flash_attn flash_attn_main.cpp flash_attn_test.cpp npu_op_test_util.cpp)
target_include_directories(op_test_flash_attn PUBLIC ${EIGEN3_INCLUDE_DIR} ${ACL_PATH}/include ../prebuild)
target_link_directories(op_test_flash_attn PUBLIC ${ACL_PATH}/lib64 ../prebuild)
target_link_libraries(op_test_flash_attn  sentencepiece boost_program_options boost_system boost_filesystem fmt ascendcl runtime npu_ops)

add_executable(op_test_gemm gemm_main.cpp gemm_test.cpp npu_op_test_util.cpp)
target_include_directories(op_test_gemm PUBLIC ${EIGEN3_INCLUDE_DIR} ${ACL_PATH}/include ../prebuild)
target_link_directories(op_test_gemm PUBLIC ${ACL_PATH}/lib64 ../prebuild)
target_link_libraries(op_test_gemm  sentencepiece boost_program_options boost_system boost_filesystem fmt ascendcl runtime npu_ops)

add_executable(op_test_rope_single rope_single_layer_main.cpp rope_single_layer_test.cpp npu_op_test_util.cpp)
target_include_directories(op_test_rope_single PUBLIC ${EIGEN3_INCLUDE_DIR} ${ACL_PATH}/include ../prebuild)
target_link_directories(op_test_rope_single PUBLIC ${ACL_PATH}/lib64 ../prebuild)
target_link_libraries(op_test_rope_single sentencepiece boost_program_options boost_system boost_filesystem fmt ascendcl runtime npu_ops)

add_executable(op_test_embedding embedding_main.cpp embedding_test.cpp npu_op_test_util.cpp)
target_include_directories(op_test_embedding PUBLIC ${EIGEN3_INCLUDE_DIR} ${ACL_PATH}/include ../prebuild)
target_link_directories(op_test_embedding PUBLIC ${ACL_PATH}/lib64 ../prebuild)
target_link_libraries(op_test_embedding sentencepiece boost_program_options boost_system boost_filesystem fmt ascendcl runtime npu_ops)

