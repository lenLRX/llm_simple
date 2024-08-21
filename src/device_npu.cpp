#include "acl/acl.h"
#include "device.hpp"
#include "acl_util.hpp"


NPUAllocator& NPUAllocator::GetInstance() {
    static NPUAllocator instance;
    return instance;
}

void* NPUAllocator::Allocate(size_t size) {
    void *dev_mem = nullptr;
    CHECK_ACL(aclrtMalloc(&dev_mem, size, ACL_MEM_MALLOC_HUGE_FIRST));
    return dev_mem;
}

void NPUAllocator::Deallocate(void* ptr) {
    CHECK_ACL(aclrtFree(ptr));
}



std::shared_ptr<Tensor> Tensor::MakeNPUTensor(size_t size, DataType dtype) {
    auto result = std::make_shared<Tensor>();
    result->data_size = size;
    result->data_ptr = NPUAllocator::Allocate(SizeOfTensor(size, dtype));
    result->data_type = dtype;
    result->dev_type = DEV_NPU;
    return result;
}