#gdb --args \
./build/src/llama2_main \
--config=llama/llama2/7B/params.json \
--tokenizer=llama/llama2/tokenizer.model \
--weight=/data/llama2/7B/model_output \ 
--device_type=npu \
--max_seq_len=2048 \
--log_level=info \
--prompt="Translate English to French:
    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>"
