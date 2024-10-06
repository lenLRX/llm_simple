#gdb --args \
./build/src/llama2_main \
--config=/data/llama2/7B/params.json \
--tokenizer=/data/llama2_7b_awq/tokenizer.model \
--weight=/data/llama2_7b_awq/llama2_7b_awq_4bit/ \
--device_type=npu \
--max_seq_len=2048 \
--max_gen_token=256 \
--log_level=info \
--debug_print=false \
--quant_method=awq_4bit \
--quant_group_size=128 \
--rope_is_neox_style=true \
--prompt="Translate English to French:
    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>"
