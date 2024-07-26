from typing import Dict
import pandas as pd
from llm.send import ChatLLM
from df_ops import summarize_descriptions
from utils.save import parquet_table_save


def create_summarized_entities(
    dataset: pd.DataFrame,
    query_output_dir: str,
    send_to: ChatLLM,
    llm_args: Dict = {},
    save: bool = True,
):
    dataset = summarize_descriptions(
        dataset,
        send_to=send_to,
        column="entity_graph",
        to="entity_graph",
        strategy={
            "llm": {
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "max_summary_length": 500,
        },
        llm_args=llm_args,
    )

    if save:
        parquet_table_save(
            query_output_dir,
            "create_summarized_entities",
            dataset,
        )

    return dataset
