#pragma once

#include <memory>

#include "defs.hpp"

class CPUAllocator {
public:
    static CPUAllocator& GetInstance();
    static void* Allocate(size_t size);
    static void Deallocate(void* ptr);
};

class NPUAllocator {
public:
    static NPUAllocator& GetInstance();
    static void* Allocate(size_t size);
    static void Deallocate(void* ptr);
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor() = default;
    ~Tensor();
    static std::shared_ptr<Tensor> MakeCPUTensor(size_t size, DataType dtype);
    static std::shared_ptr<Tensor> MakeNPUTensor(size_t size, DataType dtype);

    std::shared_ptr<Tensor> to(DeviceType to_dev);

    void* data_ptr{nullptr};
    size_t data_size;
    DataType data_type;
    DeviceType dev_type;
};

