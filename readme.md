# Orange Pi LLM推理
支持OrangePi LLM推理，当前测试硬件版本: Orange Pi 20T 24GB
# 安装
[安装文档](orangepi_install.md)
# 权重转换
## LLAMA2-7B FP16 (支持llama官方发布的格式, 包含tokenizer.model,params.json,consolidated.00.pth文件)
```python3 scripts/convert_llama2_weight.py --input_dir <llama_path> --model_size 7B --output_dir <output dir>```
## LLAMA2-AWQ 4bit
权重下载链接:[model.safetensors](https://huggingface.co/TheBloke/Llama-2-7B-AWQ/blob/main/model.safetensors)
```python3 scripts/convert_llama_awq_4bit.py --input_safetensor <model.safetensors path> --output_dir <weight output path>```

# 运行
*请将转化后的权重文件夹，配置文件, tokenizer文件拷贝到设备上并修改bash文件中对应的路径*

1. ```bash scripts/example_chat_llama2_7B_fp16_orangepi.sh```
2. ```bash scripts/example_text_completion_llama2_7B_fp16_orangepi.sh```
3. ```bash scripts/example_chat_llama2_7B_awq_4bit_orangepi.sh```
4. ```bash scripts/example_text_completion_llama2_7B_awq_4bit_orangepi.sh```

# 性能
|场景|ttft(ms)|decode(ms/token)|
|---|---|---|
|llama2-7B-AWQ-4bit|886|176.7|
|llama2-7B-FP16|4498|568.4|
