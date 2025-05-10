#include <algorithm>
#include <boost/filesystem.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>

#include "device.hpp"
#include "llama2_layer_cpu.hpp"
#include "llama2_layer_npu.hpp"
#include "llama2_model.hpp"
#include "util.h"

using namespace std::literals;

bool Llama2Model::Init() {
  if (config.device_type == DEV_NPU) {
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&model_stream));
  }
  std::ifstream config_fs(config.config_path.c_str());
  config_fs >> config.config;

  tokenizer.Init(config.tok_path);

  spdlog::info("using json config\n{}", config.config.dump(4));
  hidden_dim = config.config["dim"].get<int>();
  n_heads = config.config["n_heads"].get<int>();
  n_layers = config.config["n_layers"].get<int>();
  norm_eps = config.config["norm_eps"].get<float>();
  multiple_of = config.config["multiple_of"].get<float>();
  n_words = tokenizer.n_words;
  pad_id = tokenizer.pad_id;

  InitFreqCIS();

  embedding_layer.Init(this, (boost::filesystem::path(config.model_path) /
                              "model.embed_tokens.weight.bin")
                                 .c_str());
  causual_mask_layer.Init(this);
  transformer_layers.resize(n_layers);
  for (int i = 0; i < n_layers; ++i) {
    spdlog::info("loading layer {}/{}", i, n_layers);
    transformer_layers[i].Init(this, i);
  }

  last_norm.Init(this, -1, false, true);
  last_mm.Init(
      this,
      (boost::filesystem::path(config.model_path) / "lm_head.weight.bin")
          .c_str(),
      tokenizer.n_words, hidden_dim);
  argmax_layer.Init(this);
  top_p_layer.Init(this);

  return true;
}

static int find_first_diff_str(const std::string &lhs, const std::string &rhs) {
  int common_size = std::min(lhs.size(), rhs.size());
  for (int i = 0; i < common_size; ++i) {
    if (lhs[i] != rhs[i]) {
      return i;
    }
  }
  return common_size;
}

void Llama2Model::Chat(const std::string &input_seq,
                       const std::string &reverse_prompt) {
  auto tokens = tokenizer.Encode(input_seq, true, false);
  auto reverse_prompt_size = reverse_prompt.size();
  int input_token_size = tokens.size();
  spdlog::debug("promt token size {}", input_token_size);
  std::string prev_full_string = input_seq;

  std::cout << input_seq;

  int prev_pos = 0;
  int cur_pos = input_token_size;

  bool is_interacting = true;

  do {
    if (is_interacting) {
      std::string user_input;
      std::getline(std::cin, user_input);
      user_input = user_input + "\n";
      // spdlog::info("user_input {}", user_input);
      auto user_input_tokens = tokenizer.Encode(user_input, false, false);
      auto user_input_tokens_size = user_input_tokens.size();

      tokens.insert(tokens.end(), user_input_tokens.begin(),
                    user_input_tokens.end());
      cur_pos += user_input_tokens_size;
      prev_full_string = tokenizer.Decode(tokens);
      is_interacting = false;
    }

    if (cur_pos > config.max_seq_len) {
      std::cout << std::endl;
      spdlog::info("cur_pos {} greater than max_seq_len {}, end generation",
                   cur_pos, prev_pos);
      return;
    }

    InferenceCtx ctx(this, cur_pos, prev_pos);
    if (config.device_type == DEV_NPU) {
      ctx.npu_stream = model_stream;
    }
    spdlog::debug("cur_pos {} prev_pos {} curr size {}", cur_pos, prev_pos,
                  ctx.cur_size);
    auto input_token = Tensor::MakeCPUTensor(ctx.cur_size, DT_UINT32);
    memcpy(input_token->data_ptr, tokens.data() + prev_pos,
           sizeof(uint32_t) * ctx.cur_size);
    input_token = input_token->to(config.device_type);
    auto h = embedding_layer.Forward(input_token, ctx);

    if (config.debug_print) {
      auto print_h = h->to(DEV_CPU);
      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          embed_map(static_cast<Eigen::half *>(print_h->data_ptr), ctx.cur_size,
                    hidden_dim);

      Eigen::array<Eigen::Index, 2> offsets = {0, 0};
      Eigen::array<Eigen::Index, 2> extents = {
          static_cast<Eigen::Index>(ctx.cur_size), 4};
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
          print_slice = embed_map.slice(offsets, extents);
      std::cout << "embed output \n" << print_slice << "\n";
      // break;
    }

    auto causlmask = causual_mask_layer.Forward(ctx);

    for (int i = 0; i < n_layers; ++i) {
      h = transformer_layers[i].Forward(h, causlmask, ctx);

      if (config.debug_print) {
        auto print_h = h->to(DEV_CPU);
        Eigen::TensorMap<
            Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
            h_map(static_cast<Eigen::half *>(print_h->data_ptr), ctx.cur_size,
                  hidden_dim);
        Eigen::array<Eigen::Index, 2> offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> extents = {
            static_cast<Eigen::Index>(ctx.cur_size), 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
            print_slice = h_map.slice(offsets, extents);
        std::cout << "h_map " << i << " output \n" << print_slice << "\n";
      }
      // break;
    }
    // break;

    h = last_norm.Forward(h, ctx);

    if (config.debug_print) {
      CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
      auto print_h = h->to(DEV_CPU);
      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          h_map(static_cast<Eigen::half *>(print_h->data_ptr), ctx.cur_size,
                hidden_dim);
      Eigen::array<Eigen::Index, 2> offsets = {0, 0};
      Eigen::array<Eigen::Index, 2> extents = {
          static_cast<Eigen::Index>(ctx.cur_size), 288};
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
          print_slice = h_map.slice(offsets, extents);
      std::cout << "h_map after norm output \n" << print_slice << "\n";
    }

    h = last_mm.Forward(h, ctx);

    if (config.debug_print) {
      CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
      auto print_h = h->to(DEV_CPU);
      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          h_map(static_cast<Eigen::half *>(print_h->data_ptr), ctx.cur_size,
                tokenizer.n_words);
      Eigen::array<Eigen::Index, 2> offsets = {0, 0};
      Eigen::array<Eigen::Index, 2> extents = {
          static_cast<Eigen::Index>(ctx.cur_size), 4};
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
          print_slice = h_map.slice(offsets, extents);
      std::cout << "h_map last mm output \n" << print_slice << "\n";
    }

    if (config.device_type == DEV_NPU) {
      CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    }

    h = h->to(DEV_CPU);
    int next_tok;
    if (config.temperature > 0.0f) {
      next_tok = top_p_layer.Forward(h, ctx);
    } else {
      auto max_pos = argmax_layer.Forward(h, ctx);
      next_tok = static_cast<int32_t *>(max_pos->data_ptr)[ctx.cur_size - 1];
    }
    if (next_tok == tokenizer.eos_id) {
      is_interacting = true;
    }
    spdlog::debug("cur_pos {} max_seq_len {} next_tok {}", cur_pos, config.max_seq_len,
                  next_tok);
    tokens.push_back(next_tok);

    std::stringstream new_ss;
    auto full_string = tokenizer.Decode(tokens);
    int common_size = find_first_diff_str(full_string, prev_full_string);
    if (common_size < prev_full_string.size()) {
      // need to delete some char
      for (int b = 0; b < prev_full_string.size() - common_size; ++b) {
        new_ss << '\b';
      }
    }
    prev_full_string = full_string;
    std::string new_str = full_string.substr(common_size);
    new_ss << new_str;

    int rstring_offset = full_string.size() - reverse_prompt_size;
    if (rstring_offset >= 0) {
      bool match_reverse_prompt = true;
      for (int ri = 0; ri < reverse_prompt_size; ++ri) {
        if (full_string[rstring_offset + ri] != reverse_prompt[ri]) {
          match_reverse_prompt = false;
        }
      }

      if (match_reverse_prompt) {
        is_interacting = true;
      }
    }

    std::cout << new_ss.str() << std::flush;

    prev_pos = cur_pos;
    ++cur_pos;
  } while (cur_pos < config.max_seq_len);

  std::cout << std::endl;
}

void Llama2Model::TextCompletion(const std::string &input_seq) {
  auto tokens = tokenizer.Encode(input_seq, true, false);
  tokens.reserve(config.max_seq_len);
  int input_token_size = tokens.size();
  int generate_limit = std::min(config.max_seq_len, input_token_size + config.max_gen_len);
  spdlog::debug(
      "Llama2Model::TextCompletion input \"{}\" promt size {} token_size {}",
      input_seq, input_token_size, config.max_seq_len);
  std::string prev_full_string = input_seq;
  int output_seq_len = input_token_size;

  std::cout << "input prompt:\n" << input_seq;

  int prev_pos = 0;
  auto start_tp = std::chrono::steady_clock::now();
  float time_to_first_token_ms = -1;
  int decode_token_num = 0;
  for (int cur_pos = input_token_size; cur_pos < generate_limit; ++cur_pos) {
    spdlog::debug("cur_pos {} prev_pos {}", cur_pos, prev_pos);
    InferenceCtx ctx(this, cur_pos, prev_pos);
    if (config.device_type == DEV_NPU) {
      ctx.npu_stream = model_stream;
    }
    auto input_token = Tensor::MakeCPUTensor(ctx.cur_size, DT_UINT32);
    memcpy(input_token->data_ptr, tokens.data() + prev_pos,
           sizeof(uint32_t) * ctx.cur_size);
    input_token = input_token->to(config.device_type);
    auto h = embedding_layer.Forward(input_token, ctx);

    if (config.debug_print) {
      auto print_h = h->to(DEV_CPU);
      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          embed_map(static_cast<Eigen::half *>(print_h->data_ptr), ctx.cur_size,
                    hidden_dim);

      Eigen::array<Eigen::Index, 2> offsets = {0, 0};
      Eigen::array<Eigen::Index, 2> extents = {
          static_cast<Eigen::Index>(ctx.cur_size), 4};
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
          print_slice = embed_map.slice(offsets, extents);
      std::cout << "embed output \n" << print_slice << "\n";
      // break;
    }

    auto causlmask = causual_mask_layer.Forward(ctx);

    for (int i = 0; i < n_layers; ++i) {
      h = transformer_layers[i].Forward(h, causlmask, ctx);

      if (config.debug_print) {
        auto print_h = h->to(DEV_CPU);
        Eigen::TensorMap<
            Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
            h_map(static_cast<Eigen::half *>(print_h->data_ptr), ctx.cur_size,
                  hidden_dim);
        Eigen::array<Eigen::Index, 2> offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> extents = {
            static_cast<Eigen::Index>(ctx.cur_size), 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
            print_slice = h_map.slice(offsets, extents);
        std::cout << "h_map " << i << " output \n" << print_slice << "\n";
      }
      // break;
    }
    // break;

    h = last_norm.Forward(h, ctx);

    if (config.debug_print) {
      CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
      auto print_h = h->to(DEV_CPU);
      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          h_map(static_cast<Eigen::half *>(print_h->data_ptr), ctx.cur_size,
                hidden_dim);
      Eigen::array<Eigen::Index, 2> offsets = {0, 0};
      Eigen::array<Eigen::Index, 2> extents = {
          static_cast<Eigen::Index>(ctx.cur_size), 288};
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
          print_slice = h_map.slice(offsets, extents);
      std::cout << "h_map after norm output \n" << print_slice << "\n";
    }

    h = last_mm.Forward(h, ctx);

    if (config.debug_print) {
      CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
      auto print_h = h->to(DEV_CPU);
      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          h_map(static_cast<Eigen::half *>(print_h->data_ptr), ctx.cur_size,
                tokenizer.n_words);
      Eigen::array<Eigen::Index, 2> offsets = {0, 0};
      Eigen::array<Eigen::Index, 2> extents = {
          static_cast<Eigen::Index>(ctx.cur_size), 4};
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
          print_slice = h_map.slice(offsets, extents);
      std::cout << "h_map last mm output \n" << print_slice << "\n";
    }

    if (config.device_type == DEV_NPU) {
      CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    }

    h = h->to(DEV_CPU);
    int next_tok;
    if (config.temperature > 0.0f) {
      next_tok = top_p_layer.Forward(h, ctx);
    } else {
      auto max_pos = argmax_layer.Forward(h, ctx);
      next_tok = static_cast<int32_t *>(max_pos->data_ptr)[ctx.cur_size - 1];
    }

    if (time_to_first_token_ms < 0) {
      auto curr_tp = std::chrono::steady_clock::now();
      time_to_first_token_ms = (curr_tp - start_tp) / 1ms;
    } else {
      ++decode_token_num;
    }

    if (next_tok == tokenizer.eos_id) {
      break;
    }
    spdlog::debug("cur_pos {} max_seq_len {} next_tok {}", cur_pos, config.max_seq_len,
                  next_tok);
    tokens.push_back(next_tok);
    ++output_seq_len;

    auto output_token = tokens;
    output_token.resize(output_seq_len);
    auto next_tok_str = tokenizer.Decode(output_token);
    spdlog::debug("Llama2Model::Forward input \"{}\" output string {}",
                  input_seq, next_tok_str);

    std::stringstream new_ss;
    auto full_string = tokenizer.Decode(tokens);
    int common_size = find_first_diff_str(full_string, prev_full_string);
    if (common_size < prev_full_string.size()) {
      // need to delete some char
      for (int b = 0; b < prev_full_string.size() - common_size; ++b) {
        new_ss << '\b';
      }
    }
    prev_full_string = full_string;
    std::string new_str = full_string.substr(common_size);
    new_ss << new_str;
    std::cout << new_ss.str() << std::flush;

    prev_pos = cur_pos;
  }

  auto curr_tp = std::chrono::steady_clock::now();
  float total_time_ms = (curr_tp - start_tp) / 1ms;
  float decode_time_ms = total_time_ms - time_to_first_token_ms;

  std::cout << std::endl;

  auto output_token = tokens;
  output_token.resize(output_seq_len);

  auto next_tok_str = tokenizer.Decode(output_token);

  spdlog::debug("Llama2Model::TextCompletion input \"{}\" output string {}",
                input_seq, next_tok_str);
  spdlog::info("Llama2Model::TextCompletion total_time: {:.2f}ms",
               total_time_ms);
  spdlog::info("Llama2Model::TextCompletion total_time: prompt token {}, time "
               "to first token: {:.2f}ms",
               input_token_size, time_to_first_token_ms);
  spdlog::info("Llama2Model::TextCompletion total_time: decode stage token {}, "
               "total decoding time: {:.2f}ms, {:.2f}ms per token",
               decode_token_num, decode_time_ms,
               decode_time_ms / decode_token_num);
}

void Llama2Model::Benchmark(int input_seq_len, int output_seq_len) {
  std::vector<int> input_tokens;
  input_tokens.resize(input_seq_len, 0);
  int generate_limit = std::min(config.max_seq_len, input_seq_len + config.max_gen_len);
  spdlog::debug(
      "Llama2Model::Benchmark input seq len {} output seq len {}",
      input_seq_len, output_seq_len);

  int prev_pos = 0;
  auto start_tp = std::chrono::steady_clock::now();
  float time_to_first_token_ms = -1;
  int decode_token_num = 0;
  for (int cur_pos = input_seq_len; cur_pos < generate_limit; ++cur_pos) {
    InferenceCtx ctx(this, cur_pos, prev_pos);
    if (config.device_type == DEV_NPU) {
      ctx.npu_stream = model_stream;
    }
    auto input_token = Tensor::MakeCPUTensor(ctx.cur_size, DT_UINT32);
    memcpy(input_token->data_ptr, input_tokens.data() + prev_pos,
           sizeof(uint32_t) * ctx.cur_size);
    input_token = input_token->to(config.device_type);
    auto h = embedding_layer.Forward(input_token, ctx);

    auto causlmask = causual_mask_layer.Forward(ctx);

    for (int i = 0; i < n_layers; ++i) {
      h = transformer_layers[i].Forward(h, causlmask, ctx);
    }

    h = last_norm.Forward(h, ctx);
    h = last_mm.Forward(h, ctx);


    if (config.device_type == DEV_NPU) {
      CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    }

    h = h->to(DEV_CPU);
    int next_tok;
    if (config.temperature > 0.0f) {
      next_tok = top_p_layer.Forward(h, ctx);
    } else {
      auto max_pos = argmax_layer.Forward(h, ctx);
      next_tok = static_cast<int32_t *>(max_pos->data_ptr)[ctx.cur_size - 1];
    }

    if (time_to_first_token_ms < 0) {
      auto curr_tp = std::chrono::steady_clock::now();
      time_to_first_token_ms = (curr_tp - start_tp) / 1ms;
    } else {
      ++decode_token_num;
    }

    if (next_tok == tokenizer.eos_id) {
      break;
    }
    input_tokens.push_back(next_tok);
    ++output_seq_len;

    prev_pos = cur_pos;
  }

  auto curr_tp = std::chrono::steady_clock::now();
  float total_time_ms = (curr_tp - start_tp) / 1ms;
  float decode_time_ms = total_time_ms - time_to_first_token_ms;

  std::cout << std::endl;

  spdlog::info("Llama2Model::TextCompletion total_time: decode stage token {}, "
               "total decoding time: {:.2f}ms, {:.2f}ms per token",
               decode_token_num, decode_time_ms,
               decode_time_ms / decode_token_num);
}


std::string Llama2Model::GetCurrTokenString(size_t prev_string_size,
                                            const std::vector<int> &tokens) {
  auto full_string = tokenizer.Decode(tokens);
  return full_string.substr(prev_string_size + 1);
}
