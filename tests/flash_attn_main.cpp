#include "flash_attn_test.h"

namespace po = boost::program_options;

int main(int argc, char **argv) {
  aclrtContext context;
  int32_t deviceId{0};
  int m;
  int n;
  int offset;
  int head_num;
  int head_dim;

  po::options_description desc("GemmAWQ4BitOpTest options");
  desc.add_options()("help", "produce help message")                    //
      ("m", po::value<int>(&m)->default_value(2048), "m. default:2048") //
      ("n", po::value<int>(&n)->default_value(2048), "n. default:2048") //
      ("offset", po::value<int>(&offset)->default_value(0),
       "offset. default:0") //
      ("head_num", po::value<int>(&head_num)->default_value(32),
       "head_num. default:32") //
      ("head_dim", po::value<int>(&head_dim)->default_value(128),
       "head_dim. default:128");

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

  FlashAttentionOpTest op_test;
  op_test.Init(m, n, head_num, head_dim);
  bool test_result = op_test.Run(m, n, offset);
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
