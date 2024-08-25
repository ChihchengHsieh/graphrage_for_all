#!/bin/bash

python3 extending_features.py \
    --output_dir=./llama3_index_results --store_type=graphrag \
    --source=huggingface --model=meta-llama/Meta-Llama-3.1-8B-Instruct \
    --doc_top_k=1
