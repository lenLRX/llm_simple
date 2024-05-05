#include <spdlog/spdlog.h>

#include "tokenizer.hpp"


bool Tokenizer::Init(const std::string& token_model_path) {
    const auto status = processor.Load(token_model_path);
    if (!status.ok() ) {
        spdlog::critical("failed to init tokenizer from path {}", token_model_path);
        return false;
    }

    n_words = processor.GetPieceSize();
    bos_id = processor.bos_id();
    eos_id = processor.eos_id();
    pad_id = processor.pad_id();

    spdlog::info("initialized tokenizer from {}, nwords: {}, bos_id: {}, eos_id: {}, pad_id: {}",
        token_model_path, n_words, bos_id, eos_id, pad_id);

    return true;
}


std::vector<int32_t> Tokenizer::Encode(const std::string& text, bool bos, bool eos) {
    std::vector<int32_t> result;
    processor.Encode(text, &result);
    if (bos) {
        result.insert(result.begin(), bos_id);
    }
    if (eos) {
        result.push_back(eos_id);
    }
    return result;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids) {
    std::string result;
    processor.Decode(ids, &result);
    return result;
}
