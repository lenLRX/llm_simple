#pragma once

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

class Llama2Model : public ModelBase {
public:
  Tokenizer tokenizer;

  virtual bool Init() override;

  void Chat(const std::string &input_seq, const std::string &reverse_prompt) override;
  void TextCompletion(const std::string &input_seq) override;
  void Benchmark(int input_seq_len, int output_seq_len) override;
  std::string GetCurrTokenString(size_t prev_string_size,
                                 const std::vector<int> &tokens);

  int multiple_of;

  EmbeddingLayer embedding_layer;
  CausualMaskLayer causual_mask_layer;
  std::vector<Llamma2TransformerLayer> transformer_layers;
  RMSNormLayer last_norm;
  MatmulLayer last_mm;
  ArgMaxLayer argmax_layer;
  SampleTopPLayer top_p_layer;
};
