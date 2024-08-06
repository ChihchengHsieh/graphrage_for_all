python3 run_index.py \
    --query atelectasis \
    --store_type vectorstore \
    --output_dir ./index_results --doc_top_k 1 \
    --text_emb_source huggingface --text_emb_model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    # --source huggingface --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \