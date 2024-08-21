#include "device.hpp"
#include "acl_util.hpp"


Tensor::~Tensor() {
    switch (dev_type)
    {
    case DEV_NPU:
        NPUAllocator::Deallocate(data_ptr);
        break;
    case DEV_CPU:
        CPUAllocator::Deallocate(data_ptr);
        break;
    
    default:
        break;
    }
}


std::shared_ptr<Tensor> Tensor::to(DeviceType to_dev) {
    switch (dev_type) {
        case DEV_NPU:
            switch (to_dev) {
                case DEV_NPU:
                    return shared_from_this();
                case DEV_CPU: 
                {
                    auto result = Tensor::MakeCPUTensor(data_size, data_type);
                    CHECK_ACL(aclrtMemcpy(result->data_ptr, SizeOfTensor(data_size, data_type), data_ptr,
                            SizeOfTensor(data_size, data_type), ACL_MEMCPY_DEVICE_TO_HOST));
                    return result;
                }
            }
            break;
        case DEV_CPU:
            switch (to_dev) {
                case DEV_NPU:
                    {
                    auto result = Tensor::MakeNPUTensor(data_size, data_type);
                    CHECK_ACL(aclrtMemcpy(result->data_ptr, SizeOfTensor(data_size, data_type), data_ptr,
                            SizeOfTensor(data_size, data_type), ACL_MEMCPY_HOST_TO_DEVICE));
                    return result;
                    }
                case DEV_CPU:
                    return shared_from_this();
            }
            break;
    }
}
