#!/bin/bash

python3 extending_features.py \
    --output_dir=./gpt35t_index_results --store_type=graphrag \
    --source=openai --model=gpt-3.5-turbo \
    --doc_top_k=1
