#include "embedding_test.h"

namespace po = boost::program_options;


int main(int argc, char **argv) {
  aclrtContext context;
  int32_t deviceId{0};
  int max_index_num;
  int vocab_size;
  int hidden_dim;
  int test_size;

  // clang-format off
  po::options_description desc("RopeSingleLayer options");
  desc.add_options()
        ("help", "produce help message")
        ("max_index_num", po::value<int>(&max_index_num)->default_value(4096), "max_index_num. default:4096")
        ("hidden_dim", po::value<int>(&hidden_dim)->default_value(2048), "hidden_dim. default:2048")
        ("vocab_size", po::value<int>(&vocab_size)->default_value(151936), "vocab_size. default:151936")
        ("test_size", po::value<int>(&test_size)->default_value(32), "test_size. default:32");

  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }
  po::notify(vm);


  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(deviceId));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));

  EmbeddingOpTest op_test;
  op_test.Init(max_index_num, vocab_size, hidden_dim);
  bool test_result = op_test.Run(test_size);
  op_test.CleanUp();

  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());



  if (test_result) {
    spdlog::info("test success");
  } else {
    spdlog::error("test failed");
  }
}
