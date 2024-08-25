import os
import pandas as pd
from typing import Dict

from . import df_ops
from ..llm.send import EmbLLM
from ..utils.save import parquet_table_load, parquet_table_save

text_emb_strategy = {
    "batch_size": 16,
    "batch_max_tokens": 8191,
}


def create_final_entities(
    dataset: pd.DataFrame,
    query_output_dir: str,
    text_emb_llm_send_to: EmbLLM,
    text_emb_llm_args: Dict = {},
    save: bool = True,
    try_load: bool = True,
):

    fn_name = "create_final_entities"
    if try_load and os.path.exists(
        os.path.join(query_output_dir, f"{fn_name}.parquet")
    ):
        return parquet_table_load(query_output_dir, fn_name)
    dataset = df_ops.unpack_graph(
        dataset,
        column="clustered_graph",
        type="nodes",
    )

    dataset = df_ops.rename(
        dataset,
        columns={"label": "title"},
    )

    dataset = df_ops.select(
        dataset,
        columns=[
            "id",
            "title",
            "type",
            "description",
            "human_readable_id",
            "graph_embedding",
            "source_id",
        ],
    )

    dataset = df_ops.dedupe(dataset, columns=["id"])

    dataset = df_ops.rename(
        dataset,
        columns={"title": "name"},
    )

    dataset = df_ops.filter_verb(
        dataset,
        column="name",
        operator="is not empty",
        strategy="value",
        value=None,
    )

    dataset = df_ops.text_split(
        dataset,
        **{"separator": ",", "column": "source_id", "to": "text_unit_ids"},
    )

    dataset = df_ops.drop(dataset, columns=["source_id"])

    dataset = df_ops.merge(
        dataset,
        **{
            "strategy": "concat",
            "columns": ["name", "description"],
            "to": "name_description",
            "delimiter": ":",
            "preserveSource": True,
        },
    )

    dataset = df_ops.text_embed(
        input=dataset,
        send_to=text_emb_llm_send_to,
        llm_args=text_emb_llm_args,
        column="name_description",
        to="description_embedding",
        strategy=text_emb_strategy,
    )

    dataset = df_ops.drop(
        dataset,
        columns=["name_description"],
    )

    dataset = df_ops.filter_verb(
        chunk=dataset,
        column="description_embedding",
        operator="is not empty",
        strategy="value",
        value=None,
    )

    if save:
        parquet_table_save(query_output_dir, fn_name, dataset)
    return dataset
