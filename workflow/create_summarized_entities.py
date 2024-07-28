from typing import Dict
import pandas as pd
from llm.send import ChatLLM
import df_ops
import os
from utils.save import parquet_table_load, parquet_table_save


def create_summarized_entities(
    dataset: pd.DataFrame,
    query_output_dir: str,
    send_to: ChatLLM,
    llm_args: Dict = {},
    save: bool = True,
    try_load: bool = True,
):
    fn_name = "create_summarized_entities"
    if try_load and os.path.exists(
        os.path.join(query_output_dir, f"{fn_name}.parquet")
    ):
        return parquet_table_load(query_output_dir, fn_name)

    dataset = df_ops.summarize_descriptions(
        input=dataset,
        send_to=send_to,
        column="entity_graph",
        to="entity_graph",
        max_summary_length=500,
        llm_args=llm_args,
    )

    if save:
        parquet_table_save(query_output_dir, fn_name, dataset)

    return dataset
