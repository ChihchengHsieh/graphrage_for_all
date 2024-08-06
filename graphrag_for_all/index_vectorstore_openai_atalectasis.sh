python3 run_index.py \
    --query atelectasis \
    --store_type vectorstore \
    --output_dir ./index_results --doc_top_k 1 \
    --text_emb_source openai --text_emb_model_name text-embedding-3-small
# --source openai --model_name gpt-3.5-turbo \
