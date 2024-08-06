# rag-aug


Indexing (Generating graph context) in GraphRAG:
```
├── graphrag.index.__main__()
    ├── index_cli()
        ├── run_pipeline_with_config() # Nothing important
            ├── run_pipeline() # Nothing important 
            │   ├── workflow.run() # Nothing important
            │       ├── entity_extract() # Main function start
            │           ├── strategy_exec/run_gi() # where strategy is **run_gi()i**. It's  called strategy in datashaper,
            │               │                        (The document will be sent to this function one by one.)
            │               ├── load_llm() # the LLM is constructed here.
            │               ├── run_extract_entities() # where it actually
            │                   ├── TextSplitter # can split the input document into multiple text chunks.
            │                   ├── GraphExtractor # instance used to extract entities from the input document using LLMs.
            ├── ParquetTableEmitter 
            │   ├── emit () # save the generated graph.
            │   
            ├── FilePipelineStorage # used by emitter to store the graph, created by create_file_storage()
                ├── FilePipelineStorage.set() # control how the storage is saved. 
```

<!-- ├── └── │ -->


- [ ] Run the code line by line to make sure the code doesn't matter.
- [ ] Put create base text units into the whole function to test. 


# actually workflow


```
['create_base_text_units', 'create_base_extracted_entities', 'create_summarized_entities', 'create_base_entity_graph', 'create_final_entities', 'create_final_nodes', 'create_final_communities', 'join_text_units_to_entity_ids', 
'create_final_relationships', 'join_text_units_to_relationship_ids', 

'create_final_community_reports', 


'create_final_text_units', 'create_base_documents', 'create_final_documents']
```