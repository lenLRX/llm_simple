#include <algorithm>
#include <boost/filesystem.hpp>
#include <chrono>
#include <cmath>
#include <fmt/ranges.h>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>

#include "device.hpp"
#include "llama2_layer_cpu.hpp"
#include "llama2_layer_npu.hpp"
#include "qwen2_model.hpp"
#include "util.h"

using namespace std::literals;
using namespace std::chrono_literals;

// ===== streamer =====

auto StreamerGroup::put(const std::vector<int_least32_t> &output_ids) -> void {
  for (auto &streamer : streamers_) {
    streamer->put(output_ids);
  }
}

auto StreamerGroup::end() -> void {
  for (auto &streamer : streamers_) {
    streamer->end();
  }
}

auto TextStreamer::put(const std::vector<int> &output_ids) -> void {
  if (is_prompt_) {
    is_prompt_ = false;
    return;
  }

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};

  token_cache_.insert(token_cache_.end(), output_ids.begin(), output_ids.end());
  std::string text = tokenizer_->decode(token_cache_);
  if (text.empty()) {
    return;
  }

  std::string printable_text;
  if (text.back() == '\n') {
    // flush the cache after newline
    printable_text = text.substr(print_len_);
    token_cache_.clear();
    print_len_ = 0;
  } else if (std::find(puncts.begin(), puncts.end(), text.back()) !=
             puncts.end()) {
    // last symbol is a punctuation, hold on
  } else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
    // ends with an incomplete token, hold on
  } else {
    printable_text = text.substr(print_len_);
    print_len_ = text.size();
  }

  os_ << printable_text << std::flush;
}

auto TextStreamer::end() -> void {
  std::string text = tokenizer_->decode(token_cache_);
  os_ << text.substr(print_len_) << std::endl;
  is_prompt_ = true;
  token_cache_.clear();
  print_len_ = 0;
}

auto PerfStreamer::put(const std::vector<int> &output_ids) -> void {
  if (num_prompt_tokens_ == 0) {
    // before prompt eval
    start_us_ = get_current_us();
    num_prompt_tokens_ = output_ids.size();
  } else {
    if (num_output_tokens_ == 0) {
      // first new token
      prompt_us_ = get_current_us();
    }
    num_output_tokens_ += output_ids.size();
  }
}

auto PerfStreamer::reset() -> void {
  start_us_ = prompt_us_ = end_us_ = 0;
  num_prompt_tokens_ = num_output_tokens_ = 0;
}

auto PerfStreamer::to_string() -> std::string const {
  std::ostringstream oss;
  oss << "prompt time: " << prompt_total_time_us() / 1000.f << " ms / "
      << num_prompt_tokens() << " tokens (" << prompt_token_time_us() / 1000.f
      << " ms/token)\n"
      << "output time: " << output_total_time_us() / 1000.f << " ms / "
      << num_output_tokens() << " tokens (" << output_token_time_us() / 1000.f
      << " ms/token)\n"
      << "total time: "
      << (prompt_total_time_us() + output_total_time_us()) / 1000.f << " ms";
  return oss.str();
}

bool Qwen2Model::Init() {
  // if awq it should be fp16
  // config.data_type = DT_BFLOAT16;
  if (config.device_type == DEV_NPU) {
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&model_stream));
  }
  std::ifstream config_fs(config.config_path.c_str());
  config_fs >> config.config;

  spdlog::info("using json config\n{}", config.config.dump(4));
  tie_word_embeddings = config.config["tie_word_embeddings"].get<bool>();
  hidden_dim = config.config["hidden_size"].get<int>();
  n_heads = config.config["num_attention_heads"].get<int>();
  head_dim = hidden_dim / n_heads;
  n_kv_heads = config.config["num_key_value_heads"].get<int>();
  n_layers = config.config["num_hidden_layers"].get<int>();
  norm_eps = config.config["rms_norm_eps"].get<float>();
  intermediate_size = config.config["intermediate_size"].get<int>();

  qwen_tokenizer.from_pretrained(config.tok_path);

  n_words = config.config["vocab_size"].get<int>();
  pad_id = qwen_tokenizer.eos_token_id;

  InitFreqCIS(config.config["rope_theta"].get<float>(), config.data_type);

  // clang-format off
  embedding_layer.Init(this, (boost::filesystem::path(config.model_path) /
                              "model.embed_tokens.weight.bin").c_str());
  causual_mask_layer.Init(this);
  transformer_layers.resize(n_layers);
  for (int i = 0; i < n_layers; ++i) {
    spdlog::info("loading layer {}/{}", i, n_layers);
    transformer_layers[i].Init(this, i);
  }

  last_norm.Init(this, "model.norm.weight.bin");
  last_mm.Init(
      this,
      tie_word_embeddings ? (boost::filesystem::path(config.model_path) / "model.embed_tokens.weight.bin").c_str(): 
        (boost::filesystem::path(config.model_path) / "lm_head.weight.bin").c_str(),
      n_words, hidden_dim);
  // clang-format on
  argmax_layer.Init(this);
  top_p_layer.Init(this);

  return true;
}

static auto get_utf8_line(std::string &line) -> bool {
  return !!std::getline(std::cin, line);
}

void Qwen2Model::Chat(const std::string &input_seq,
                      const std::string &reverse_prompt) {
  std::vector<std::string> history;

  while (1) {
    std::cout << std::setw(config.model_type.size()) << std::left << "Prompt"
              << " > " << std::flush;
    std::string prompt;
    if (!get_utf8_line(prompt)) {
      break;
    }
    if (prompt.empty()) {
      continue;
    }
    history.emplace_back(std::move(prompt));
    std::cout << config.model_type << " > ";
    InferenceCtx ctx(this, 0, 0);
    ctx.npu_stream = model_stream;

    std::string output = Generate(history, ctx, nullptr);
    std::cout << output << std::endl;
    history.emplace_back(std::move(output));
  }
  std::cout << "Bye\n";
}

void Qwen2Model::TextCompletion(const std::string &input_seq) {
  auto ts = std::make_shared<TextStreamer>(std::cout, &qwen_tokenizer);
  auto ps = std::make_shared<PerfStreamer>();
  StreamerGroup sg({ts, ps});
  InferenceCtx ctx(this, 0, 0);
  ctx.npu_stream = model_stream;
  std::vector<int> input_ids =
      qwen_tokenizer.encode(input_seq, config.max_gen_len);
  spdlog::debug("TextCompletion input ids: {}", fmt::join(input_ids, ","));
  generate_limit = std::min((int)config.max_seq_len,
                            (int)(config.max_gen_len + input_ids.size()));
  std::cout << input_seq;
  Generate(input_ids, ctx, &sg);
  std::cout << std::flush;
  spdlog::info("{}", ps->to_string());
}

void Qwen2Model::Benchmark(int input_seq_len, int output_seq_len) {
  PerfStreamer ps;
  InferenceCtx ctx(this, 0, 0);
  ctx.npu_stream = model_stream;

  generate_limit = std::min((int)config.max_seq_len,
                            (int)(input_seq_len + output_seq_len));
  
  std::vector<int> output_ids;
  output_ids.resize(input_seq_len, 0);
  ps.put(output_ids);

  int n_past = 0;

  while ((int)output_ids.size() < generate_limit) {
    ctx.cur_pos = output_ids.size();
    ctx.cur_size = ctx.cur_pos - ctx.prev_pos;
    auto next_token_id = GenerateNextToken(output_ids, ctx, n_past);
    spdlog::debug("Generated token id: {}", next_token_id);
    ctx.prev_pos = ctx.cur_pos;

    n_past = output_ids.size();
    output_ids.emplace_back(next_token_id);

    ps.put({next_token_id});
  }

  ps.end();

  spdlog::info("Benchmark result:\n{}", ps.to_string());
}


int Qwen2Model::GenerateNextToken(const std::vector<int32_t> &input_ids,
                                  InferenceCtx &ctx, int n_past) {
  int curr_input_ids_size = input_ids.size() - n_past;
  std::cout << std::fixed << std::setprecision(7);
  spdlog::debug("cur_pos {} prev_pos {} curr_size {} n_past {} device_type {}",
                ctx.cur_pos, ctx.prev_pos, ctx.cur_size, n_past,
                config.device_type);
  auto input_token = Tensor::MakeCPUTensor(ctx.cur_size, DT_UINT32);
  memcpy(input_token->data_ptr, input_ids.data() + n_past,
         sizeof(uint32_t) * ctx.cur_size);
  for (int i = 0; i < ctx.cur_size; ++i) {
    spdlog::debug("input_ids {} {}", i,
                  ((int32_t *)(input_token->data_ptr))[i]);
  }
  input_token = input_token->to(config.device_type);
  auto h = embedding_layer.Forward(input_token, ctx);

  if (config.debug_print) {
    auto print_h = h->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::bfloat16, 2, Eigen::RowMajor | Eigen::DontAlign>>
        embed_map(static_cast<Eigen::bfloat16 *>(print_h->data_ptr),
                  ctx.cur_size, hidden_dim);

    Eigen::array<Eigen::Index, 2> offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 16};
    Eigen::Tensor<Eigen::bfloat16, 2, Eigen::RowMajor | Eigen::DontAlign>
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
          Eigen::Tensor<Eigen::bfloat16, 2, Eigen::RowMajor | Eigen::DontAlign>>
          h_map(static_cast<Eigen::bfloat16 *>(print_h->data_ptr), ctx.cur_size,
                hidden_dim);
      Eigen::array<Eigen::Index, 2> offsets = {0, 0};
      Eigen::array<Eigen::Index, 2> extents = {
          static_cast<Eigen::Index>(ctx.cur_size), 4};
      Eigen::Tensor<Eigen::bfloat16, 2, Eigen::RowMajor | Eigen::DontAlign>
          print_slice = h_map.slice(offsets, extents);
      std::cout << "h_map " << i << " output \n" << print_slice << "\n";
    }
  }

  h = last_norm.Forward(h, ctx);

  if (config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto print_h = h->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::bfloat16, 2, Eigen::RowMajor | Eigen::DontAlign>>
        h_map(static_cast<Eigen::bfloat16 *>(print_h->data_ptr), ctx.cur_size,
              hidden_dim);
    Eigen::array<Eigen::Index, 2> offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::bfloat16, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = h_map.slice(offsets, extents);
    std::cout << "h_map after norm output \n" << print_slice << "\n";
  }

  h = last_mm.Forward(h, ctx);

  if (config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto print_h = h->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::bfloat16, 2, Eigen::RowMajor | Eigen::DontAlign>>
        h_map(static_cast<Eigen::bfloat16 *>(print_h->data_ptr), ctx.cur_size,
              n_words);
    Eigen::array<Eigen::Index, 2> offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::bfloat16, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = h_map.slice(offsets, extents);
    std::cout << "h_map last mm output \n" << print_slice << "\n";
  }

  CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

  h = h->to(DEV_CPU);
  int next_tok;
  if (config.temperature > 0.0f) {
    next_tok = top_p_layer.Forward(h, ctx);
  } else {
    auto max_pos = argmax_layer.Forward(h, ctx);
    next_tok = static_cast<int32_t *>(max_pos->data_ptr)[ctx.cur_size - 1];
  }
  return next_tok;
}

std::vector<int> Qwen2Model::Generate(const std::vector<int> &input_ids,
                                      InferenceCtx &ctx,
                                      BaseStreamer *streamer) {

  generate_limit = std::min((int)config.max_seq_len,
                            (int)(config.max_gen_len + input_ids.size()));
  std::vector<int> output_ids;
  output_ids.reserve(generate_limit);
  output_ids = input_ids;
  if (streamer) {
    streamer->put(input_ids);
  }

  int n_past = 0;

  while ((int)output_ids.size() < generate_limit) {
    ctx.cur_pos = output_ids.size();
    ctx.cur_size = ctx.cur_pos - ctx.prev_pos;
    auto next_token_id = GenerateNextToken(output_ids, ctx, n_past);
    spdlog::debug("Generated token id: {}", next_token_id);
    ctx.prev_pos = ctx.cur_pos;

    n_past = output_ids.size();
    output_ids.emplace_back(next_token_id);

    if (streamer) {
      streamer->put({next_token_id});
    }

    if (next_token_id == qwen_tokenizer.eos_token_id ||
        next_token_id == qwen_tokenizer.im_start_id ||
        next_token_id == qwen_tokenizer.im_end_id) {
      break;
    }
    // std::this_thread::sleep_for(1000ms);
  }

  if (streamer) {
    streamer->end();
  }

  return output_ids;
}

std::string Qwen2Model::Generate(const std::vector<std::string> &history,
                                 InferenceCtx &ctx, BaseStreamer *streamer) {
  std::vector<int> input_ids =
      qwen_tokenizer.encode_history(history, generate_limit);
  std::vector<int> output_ids = Generate(input_ids, ctx, streamer);
  std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(),
                                  output_ids.end());
  new_output_ids.erase(std::remove_if(new_output_ids.begin(),
                                      new_output_ids.end(),
                                      [this](int id) {
                                        return qwen_tokenizer.is_special_id(id);
                                      }),
                       new_output_ids.end());
  std::string output = qwen_tokenizer.decode(new_output_ids);
  return output;
}
