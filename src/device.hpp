#pragma once

#include <memory>


enum DeviceType {
    DEV_CPU,
    DEV_GPU
};

enum DataType {
    DT_UINT8,
    DT_UINT32,
    DT_FLOAT16,
    DT_FLOAT32,
};


inline static size_t SizeOfTensor(size_t size, DataType dt) {
    switch (dt)
    {
    case DT_UINT8:
        return size * sizeof(uint8_t);
        break;
    case DT_UINT32:
        return size * sizeof(uint32_t);
    case DT_FLOAT16:
        return size * sizeof(uint16_t);
    case DT_FLOAT32:
        return size * sizeof(uint32_t);
    default:
        break;
    }
    return -1;
}


class CPUAllocator {
public:
    static CPUAllocator& GetInstance();
    static void* Allocate(size_t size);
    static void Deallocate(void* ptr);
};


class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor() = default;
    ~Tensor();
    static std::shared_ptr<Tensor> MakeCPUTensor(size_t size, DataType dtype);
    void* data_ptr{nullptr};
    size_t data_size;
    DataType data_type;
    DeviceType dev_type;
};

