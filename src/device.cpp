#include "device.hpp"


Tensor::~Tensor() {
    switch (dev_type)
    {
    case DEV_CPU:
        CPUAllocator::Deallocate(data_ptr);
        break;
    
    default:
        break;
    }
}

