import df_ops
import pandas as pd
import os
from utils.save import parquet_table_load, parquet_table_save
from typing import Dict

DEFAULT_CLUSTERING_STRATEGY = {"type": "leiden", "max_cluster_size": 10}


def create_base_entity_graph(
    dataset: pd.DataFrame,
    query_output_dir: str,
    clustering_strategy: Dict = DEFAULT_CLUSTERING_STRATEGY,
    save: bool = True,
    try_load: bool = True,
):

    fn_name = "create_base_entity_graph"
    if try_load and os.path.exists(
        os.path.join(query_output_dir, f"{fn_name}.parquet")
    ):
        return parquet_table_load(query_output_dir, fn_name)
    dataset = df_ops.cluster_graph(
        dataset,
        strategy=clustering_strategy,
        column="entity_graph",
        to="clustered_graph",
        level_to="level",
    )

    dataset = df_ops.select(
        dataset,
        columns=["level", "clustered_graph"],
    )

    if save:
        parquet_table_save(query_output_dir, fn_name, dataset)

    return dataset
