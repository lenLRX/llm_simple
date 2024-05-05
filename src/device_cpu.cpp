#include "device.hpp"


CPUAllocator& CPUAllocator::GetInstance() {
    static CPUAllocator instance;
    return instance;
}

void* CPUAllocator::Allocate(size_t size) {
    return malloc(size);
}

void CPUAllocator::Deallocate(void* ptr) {
    free(ptr);
}



std::shared_ptr<Tensor> Tensor::MakeCPUTensor(size_t size, DataType dtype) {
    auto result = std::make_shared<Tensor>();
    result->data_size = size;
    result->data_ptr = CPUAllocator::Allocate(SizeOfTensor(size, dtype));
    result->data_type = dtype;
    result->dev_type = DEV_CPU;
    return result;
}
