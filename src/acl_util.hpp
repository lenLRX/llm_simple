#pragma once

#include "acl/acl.h"
#include <spdlog/spdlog.h>

#define CHECK_ACL(x)                                                           \
  do {                                                                         \
    aclError __ret = x;                                                        \
    if (__ret != ACL_ERROR_NONE) {                                             \
      spdlog::error("{}:{} aclError: {}", __FILE__, __LINE__, __ret);          \
    }                                                                          \
  } while (0);


