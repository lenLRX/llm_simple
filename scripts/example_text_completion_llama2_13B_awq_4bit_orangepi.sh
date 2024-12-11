#gdb --args \
./build/src/llama2_main \
--config=llama/llama2/13B/params.json \
--tokenizer=llama/llama2/tokenizer.model \
--weight=/data/Llama-2-13B-AWQ/llama2_13B_awq_4bit \
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
