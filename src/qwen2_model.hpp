#pragma once

#include <chrono>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "acl_util.hpp"
#include "device.hpp"
#include "model_base.hpp"
#include "profiling.hpp"
#include "tokenizer.hpp"
#include "util.h"

// qwen.cpp/qwen.h
class BaseStreamer {
public:
  virtual ~BaseStreamer() = default;
  virtual auto put(const std::vector<int> &output_ids) -> void = 0;
  virtual auto end() -> void = 0;
};

class StreamerGroup : public BaseStreamer {
public:
  StreamerGroup(std::vector<std::shared_ptr<BaseStreamer>> streamers)
      : streamers_(std::move(streamers)) {}
  auto put(const std::vector<int> &output_ids) -> void override;
  auto end() -> void override;

private:
  std::vector<std::shared_ptr<BaseStreamer>> streamers_;
};

class TextStreamer : public BaseStreamer {
public:
  TextStreamer(std::ostream &os, Qwen2HFTokenizer *tokenizer)
      : os_(os), tokenizer_(tokenizer), is_prompt_(true), print_len_(0) {}
  auto put(const std::vector<int> &output_ids) -> void override;
  auto end() -> void override;

private:
  std::ostream &os_;
  Qwen2HFTokenizer *tokenizer_;
  bool is_prompt_;
  std::vector<int> token_cache_;
  int print_len_;
};

class PerfStreamer : public BaseStreamer {
public:
  PerfStreamer()
      : start_us_(0), prompt_us_(0), end_us_(0), num_prompt_tokens_(0),
        num_output_tokens_(0) {}

  auto put(const std::vector<int> &output_ids) -> void override;
  auto end() -> void override { end_us_ = get_current_us(); }

  auto reset() -> void;
  auto to_string() -> std::string const;

  auto num_prompt_tokens() const -> int64_t { return num_prompt_tokens_; }
  auto prompt_total_time_us() const -> int64_t {
    return prompt_us_ - start_us_;
  }
  auto prompt_token_time_us() const -> int64_t {
    return num_prompt_tokens() ? prompt_total_time_us() / num_prompt_tokens()
                               : 0;
  }
  auto num_output_tokens() const -> int64_t { return num_output_tokens_; }
  auto output_total_time_us() const -> int64_t { return end_us_ - prompt_us_; }
  auto output_token_time_us() const -> int64_t {
    return num_output_tokens() ? output_total_time_us() / num_output_tokens()
                               : 0;
  }

private:
  int64_t start_us_;
  int64_t prompt_us_;
  int64_t end_us_;
  int64_t num_prompt_tokens_;
  int64_t num_output_tokens_;
};

class Qwen2Model : public ModelBase {
public:
  //QwenTokenizer qwen_tokenizer;
  Qwen2HFTokenizer qwen_tokenizer;

  virtual bool Init() override;

  void Chat(const std::string &input_seq,
            const std::string &reverse_prompt) override;
  void TextCompletion(const std::string &input_seq) override;
  void Benchmark(int input_seq_len, int output_seq_len) override;
  int GenerateNextToken(const std::vector<int32_t> &input_ids,
                        InferenceCtx &ctx, int n_past);
  std::vector<int> Generate(const std::vector<int> &input_tokens,
                            InferenceCtx &ctx, BaseStreamer *streamer);
  std::string Generate(const std::vector<std::string> &history,
                       InferenceCtx &ctx, BaseStreamer *streamer);

  bool tie_word_embeddings;
  int intermediate_size;
  int generate_limit;
  EmbeddingLayer embedding_layer;
  CausualMaskLayer causual_mask_layer;
  std::vector<Qwen2TransformerLayer> transformer_layers;
  RMSNormLayer last_norm;
  MatmulLayer last_mm;
  ArgMaxLayer argmax_layer;
  SampleTopPLayer top_p_layer;
};
