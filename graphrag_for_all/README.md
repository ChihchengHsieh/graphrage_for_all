# rag-aug


## Get started

First to install the dependency:

If pipreqs hasn't been installed yet : `pip install pipreqs`

Then run:
```
pip install -r requirements.txt
```


To use GraphRAG, following two steps are required:


1. Create a file *secret.py* with the following in information:
```python

# Huggingface API key
HUGGINGFACE_TOKEN = <YOUR HIGGING FACE TOKEN>

# OpenAI API key
OPENAI_API_KEY = <YOUR OPEN AI KEY>
```

If you choose Huggingface as the source, you need to have a Huggingface account and get the API key from the account. You will need to request access to the LLM model that you want to use.

2. Generating the knowledge graph using `run_index.py`. For example:
```bash
python3 run_index.py \
    --query atelectasis \
    --store_type graphrag \
    --output_dir ./index_results --doc_top_k 1 \
    --source huggingface --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --text_emb_source huggingface --text_emb_model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
```
The Examples for using huggingface model is shown in `index_llama3v1_atelectasis.sh`, where llama 3.1 is used. To change the entity types for LLMs to capture, change `DEFAULT_ENTITY_TYPES` in `./df_ops/defaults`, which is set as `["disease", "symptoms", "cause"]` by default.

3. Querying the knoweledge graph (communities). Notebook `run_search.ipynb` provide an example of querying using llama 3.1.

