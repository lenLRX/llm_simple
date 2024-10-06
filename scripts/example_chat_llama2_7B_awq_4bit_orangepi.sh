PROMPT_TEMPLATE=./prompts/chat_example.txt
PROMPT_FILE=$(mktemp -t llamacpp_prompt.XXXXXXX.txt)

DATE_TIME=$(date +%H:%M)
DATE_YEAR=$(date +%Y)
USER_NAME="user"
AI_NAME="orange_pi"

sed -e "s/\[\[USER_NAME\]\]/$USER_NAME/g" \
    -e "s/\[\[AI_NAME\]\]/$AI_NAME/g" \
    -e "s/\[\[DATE_TIME\]\]/$DATE_TIME/g" \
    -e "s/\[\[DATE_YEAR\]\]/$DATE_YEAR/g" \
     $PROMPT_TEMPLATE > $PROMPT_FILE

./build/src/llama2_main \
--config=/data/llama2/7B/params.json \
--tokenizer=/data/llama2/tokenizer.model \
--weight=/data/llama2_7b_awq/llama2_7b_awq_4bit/  \
--quant_method=awq_4bit \
--quant_group_size=128 \
--device_type=npu \
--max_seq_len=2048 \
--log_level=info \
--rope_is_neox_style=true \
--reverse_promt="${USER_NAME}:" \
--i \
--prompt_file=$PROMPT_FILE