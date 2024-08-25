#include "acl/acl.h"
#include <spdlog/spdlog.h>

#include "device.hpp"
#include "acl_util.hpp"


NPUAllocatorEntry::NPUAllocatorEntry(size_t s, void* p): size(s), ptr(p) {

}

bool NPUAllocatorEntry::operator < (const NPUAllocatorEntry& other){
    return this->size < other.size;
}


NPUAllocator& NPUAllocator::GetInstance() {
    static NPUAllocator instance;
    return instance;
}

void* NPUAllocator::Allocate(size_t size) {
    return GetInstance().AllocateImpl(size);
}

void* NPUAllocator::AllocateImpl(size_t size) {
    void *dev_mem = nullptr;
    for (auto it = freelist.begin(); it != freelist.end();) {
        if (it->size >= size) {
            dev_mem = it->ptr;
            it = freelist.erase(it);
            break;
        }
        else {
            ++it;
        }
    }
    if (dev_mem != nullptr) {
        return dev_mem;
    }
    CHECK_ACL(aclrtMalloc(&dev_mem, size, ACL_MEM_MALLOC_HUGE_FIRST));
    allocated_bytes+=size;
    ptr_size[dev_mem] = size;

    for (auto it = freelist.begin(); it != freelist.end() && allocated_bytes > dev_mem_max;) {
        CHECK_ACL(aclrtFree(it->ptr));
        ptr_size.erase(it->ptr);
        it = freelist.erase(it);
    }

    return dev_mem;
}

void NPUAllocator::Deallocate(void* ptr) {
    GetInstance().DeallocateImpl(ptr);
}

void NPUAllocator::DeallocateImpl(void* ptr) {
    freelist.emplace_back(ptr_size.at(ptr), ptr);
    freelist.sort();
}



std::shared_ptr<Tensor> Tensor::MakeNPUTensor(size_t size, DataType dtype) {
    auto result = std::make_shared<Tensor>();
    result->data_size = size;
    result->data_ptr = NPUAllocator::Allocate(SizeOfTensor(size, dtype));
    result->data_type = dtype;
    result->dev_type = DEV_NPU;
    return result;
}