#include "rope_single_layer_test.h"

namespace po = boost::program_options;

template <typename EigenTy> bool TestFn(size_t max_seq_length, size_t head_dim,
                                        size_t head_num, size_t seq_len, size_t offset, bool is_neox) {
  aclrtContext context;
  int32_t deviceId{0};

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(deviceId));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));

  RoPESingleOpTest<EigenTy> op_test;
  op_test.Init(max_seq_length, head_dim, head_num, is_neox);
  bool test_result = op_test.Run(offset, seq_len);
  op_test.CleanUp();

  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
  return test_result;
}

int main(int argc, char **argv) {
  aclrtContext context;
  int32_t deviceId{0};
  int max_seq_length;
  int head_dim;
  int head_num;
  int seq_len;
  int offset;
  bool is_neox;
  std::string dtype_str;

  // clang-format off
  po::options_description desc("RopeSingleLayer options");
  desc.add_options()
        ("help", "produce help message")
        ("max_seq_length", po::value<int>(&max_seq_length)->default_value(4096), "max_seq_length. default:4096")
        ("head_dim", po::value<int>(&head_dim)->default_value(128), "head_dim. default:128")
        ("head_num", po::value<int>(&head_num)->default_value(32), "head_num. default:32")
        ("seq_len", po::value<int>(&seq_len)->default_value(32), "seq_len. default:32")
        ("is_neox", po::value<bool>(&is_neox)->default_value(true), "is_neox, default:true")
        ("offset", po::value<int>(&offset)->default_value(0), "offset, default:0")
        ("dtype", po::value<std::string>(&dtype_str)->required(), "dtype. float16 or bfloat16");

  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }
  po::notify(vm);

  bool test_result = false;

  if (dtype_str == "float16") {
    test_result = TestFn<Eigen::half>(max_seq_length, head_dim, head_num, seq_len, offset, is_neox);
  } else if (dtype_str == "bfloat16") {
    test_result = TestFn<Eigen::bfloat16>(max_seq_length, head_dim, head_num, seq_len, offset, is_neox);;
  }

 
  if (test_result) {
    spdlog::info("test success");
  } else {
    spdlog::error("test failed");
  }
}
