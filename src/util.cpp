#include <fstream>
#include <spdlog/spdlog.h>

#include "util.h"

bool LoadBinaryFile(const char* path, void* buffer, size_t data_size) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        spdlog::critical("failed to open {}", path);
        return false;
    }

    ifs.read((char*)buffer, data_size);

    if (ifs.fail()) {
        spdlog::critical("failed to read {} of size {}", path, data_size);
        return false;
    }

    spdlog::debug("loaded binary {} size {}", path, data_size);

    return true;
}