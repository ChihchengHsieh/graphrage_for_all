# rag-aug


## Get started

To use GraphRAG, following two steps are required:

1. Generating the knowledge graph using `run_index.py`. For example:
```bash
python3 run_index.py \
    --query atelectasis \
    --output_dir ./index_results --doc_top_k 1 \
    --source huggingface --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --text_emb_model_name meta-llama/Meta-Llama-3.1-8B-Instruct
```
The Examples for using huggingface model is shown in `index_llama3v1_atelectasis.sh`, where llama 3.1 is used. To change the entity types for LLMs to capture, change `DEFAULT_ENTITY_TYPES` in `./df_ops/defaults`, which is set as `["disease", "symptoms", "cause"]` by default.

2. Querying the knoweledge graph (communities). Notebook `run_search.ipynb` provide an example of querying using llama 3.1.



