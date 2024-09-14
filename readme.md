# Orange Pi LLM推理
支持OrangePi LLM推理，当前测试硬件版本: Orange Pi 20T 24GB
# 安装
[安装文档](orangepi_install.md)
# 权重转换
```python3 scripts/convert_llama2_weight.py --input_dir <llama_path> --model_size 7B --output_dir <output dir>```
# 运行
*请将转化后的权重文件夹，配置文件, tokenizer文件拷贝到设备上并修改bash文件中对应的路径*

1. ```bash scripts/example_chat_llama2_7B_fp16_orangepi.sh```
2. ```bash scripts/example_text_completion_llama2_7B_fp16_orangepi.sh```