#gdb --args \
./build/src/llama2_main \
--config=/data/llama2/7B/params.json \
--tokenizer=/data/llama2/tokenizer.model \
--weight=/data/llama2/7B/model_output \
--device_type=npu \
--max_seq_len=2048 \
--log_level=info \
--prompt="Please write a story of a poor little girl."