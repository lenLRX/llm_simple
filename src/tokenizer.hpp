#pragma once

#include <sentencepiece_processor.h>

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;
    bool Init(const std::string& token_model_path);
    std::vector<int32_t> Encode(const std::string& text, bool bos, bool eos);
    std::string Decode(const std::vector<int32_t>& ids);
//private:
    sentencepiece::SentencePieceProcessor processor;

    int32_t n_words;
    int32_t bos_id;
    int32_t eos_id;
    int32_t pad_id;

};
