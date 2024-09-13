#pragma once


enum DeviceType {
    DEV_CPU,
    DEV_GPU,
    DEV_NPU
};

enum DataType {
    DT_UINT8,
    DT_INT8,
    DT_UINT32,
    DT_INT32,
    DT_FLOAT16,
    DT_FLOAT32,
};


inline static size_t SizeOfTensor(size_t size, DataType dt) {
    switch (dt)
    {
    case DT_INT8:
    case DT_UINT8:
        return size * sizeof(uint8_t);
        break;
    case DT_INT32:
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

