#pragma once

#include <memory>
#include <list>
#include <unordered_map>

#include "defs.hpp"

class CPUAllocator {
public:
    static CPUAllocator& GetInstance();
    static void* Allocate(size_t size);
    static void Deallocate(void* ptr);
};

class NPUAllocatorEntry {
public:
    NPUAllocatorEntry(size_t s, void* p);
    bool operator < (const NPUAllocatorEntry& other);
    size_t size;
    void* ptr;
};

class NPUAllocator {
public:
    static NPUAllocator& GetInstance();
    static void* Allocate(size_t size);
    static void Deallocate(void* ptr);
private:
    void* AllocateImpl(size_t size);
    void DeallocateImpl(void* ptr);

    std::list<NPUAllocatorEntry> freelist;
    std::unordered_map<void*, size_t> ptr_size;
    size_t dev_mem_max{8*1024ULL*1024ULL*1024ULL};
    size_t dev_mem_max_entry_num{128};
    size_t max_record{0};
    size_t allocated_bytes{0};
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor() = default;
    ~Tensor();
    static std::shared_ptr<Tensor> MakeCPUTensor(size_t size, DataType dtype);
    static std::shared_ptr<Tensor> MakeNPUTensor(size_t size, DataType dtype);

    std::shared_ptr<Tensor> to(DeviceType to_dev);
    void to_file(const char* path);

    void* data_ptr{nullptr};
    size_t data_size;
    DataType data_type;
    DeviceType dev_type;
};

