import os
import pandas as pd
from typing import Dict

from ..llm.send import ChatLLM
from ..template.graph_extract import GRAPH_EXTRACTION_PROMPT
from ..utils.save import parquet_table_load, parquet_table_save
from ..df_ops.entity_extract import entity_extract
from ..df_ops.merge_graphs import merge_graphs


def create_base_extracted_entities(
    dataset: pd.DataFrame,
    query_output_dir: str,
    llm_send_to: ChatLLM,
    llm_args: Dict = {},
    save: bool = True,
    try_load: bool = True,
):
    fn_name = "create_base_extracted_entities"
    if try_load and os.path.exists(
        os.path.join(query_output_dir, f"{fn_name}.parquet")
    ):
        return parquet_table_load(query_output_dir, fn_name)

    dataset = entity_extract(
        input=dataset,
        send_to=llm_send_to,
        column="chunk",
        id_column="chunk_id",
        graph_to="entity_graph",
        to="entities",
        entity_types=["disease", "symptom", "cause"],
        extraction_prompt=GRAPH_EXTRACTION_PROMPT,
        llm_args=llm_args,
    )

    dataset = merge_graphs(
        dataset,
        column="entity_graph",
        to="entity_graph",
        **{
            "nodes": {
                "source_id": {
                    "operation": "concat",
                    "delimiter": ", ",
                    "distinct": True,
                },
                "description": (
                    {
                        "operation": "concat",
                        "separator": "\n",
                        "distinct": False,
                    }
                ),
            },
            "edges": {
                "source_id": {
                    "operation": "concat",
                    "delimiter": ", ",
                    "distinct": True,
                },
                "description": (
                    {
                        "operation": "concat",
                        "separator": "\n",
                        "distinct": False,
                    }
                ),
                "weight": "sum",
            },
        },
    )

    if save:
        parquet_table_save(query_output_dir, fn_name, dataset)
    return dataset
