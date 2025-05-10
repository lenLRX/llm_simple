#pragma once

#include "tiktoken.h"

#include <sentencepiece_processor.h>
#include <nlohmann/json.hpp>
#include <Python.h>

class Tokenizer {
public:
  Tokenizer() = default;
  ~Tokenizer() = default;
  bool Init(const std::string &token_model_path);
  std::vector<int32_t> Encode(const std::string &text, bool bos, bool eos);
  std::string Decode(const std::vector<int32_t> &ids);
  // private:
  sentencepiece::SentencePieceProcessor processor;

  int32_t n_words;
  int32_t bos_id;
  int32_t eos_id;
  int32_t pad_id;
};

// from https://github.com/QwenLM/qwen.cpp/blob/master/qwen.h
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

struct QwenConfig {
  // common attributes
  // ggml_type dtype;
  int vocab_size;
  int hidden_size;
  int num_attention_heads;
  int num_kv_heads;
  int num_hidden_layers;
  int intermediate_size;
  // for sequence generation
  int max_length;
  // for tokenizer
  int eos_token_id;
  int pad_token_id;
  int im_start_id;
  int im_end_id;
};

class QwenTokenizer {
public:
  QwenTokenizer() = default;
  ~QwenTokenizer() = default;
  void Init(const std::string &tiktoken_path);

  auto encode(const std::string &text, int max_length) const
      -> std::vector<int>;

  auto decode(const std::vector<int> &ids) const -> std::string;

  auto encode_history(const std::vector<std::string> &history,
                      int max_length) const -> std::vector<int>;

  auto build_prompt(const std::vector<std::string> &history) const
      -> std::string;

  auto is_special_id(int id) const -> bool;

  tiktoken::tiktoken tokenizer;
  int eos_token_id;
  int im_start_id;
  int im_end_id;
};

class Qwen2HFTokenizer {
public:
  Qwen2HFTokenizer() = default;
  void from_pretrained(const std::string &tokenizer_dir);
  std::vector<int> encode(const std::string &text, int max_length = -1) const;
  std::string decode(const std::vector<int> &ids);

  auto encode_history(const std::vector<std::string> &history,
                      int max_length) const -> std::vector<int>;

  auto build_prompt(const std::vector<std::string> &history) const
      -> std::string;

  auto is_special_id(int id) const -> bool;

  int eos_token_id;
  int im_start_id;
  int im_end_id;

private:
  PyObject *py_tokenizer{nullptr};
  PyObject *py_tokenizer_clz{nullptr};
  PyObject *py_transformers_module{nullptr};
  PyObject *py_encode_func{nullptr};
  PyObject *py_decode_func{nullptr};
};
