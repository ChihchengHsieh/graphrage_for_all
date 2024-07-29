python3 run_index.py \
    --query atelectasis \
    --output_dir ./index_results --doc_top_k 1 \
    --source openai --model_name gpt-3.5-turbo \
    --text_emb_model_name text-embedding-3-small
