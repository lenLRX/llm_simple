#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <sstream>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>


bool LoadBinaryFile(const char* path, void* buffer, size_t data_size);

template <typename TensorType>
std::string TensorShapeToStr(TensorType& tensor) {
    auto& d = tensor.dimensions();
    std::stringstream ss;
    ss << "(";
    for (int i = 0; i < d.size(); ++i) {
        ss << d[i];
        if (i < d.size() - 1) {
            ss << ",";
        }
    }
    ss << ")";
    return ss.str();
}

