import df_ops
import pandas as pd

from typing import Dict
from utils.save import parquet_table_save

DEFAULT_CLUSTERING_STRATEGY = {"type": "leiden", "max_cluster_size": 10}


def create_base_entity_graph(
    dataset: pd.DataFrame,
    query_output_dir: str,
    clustering_strategy: Dict = DEFAULT_CLUSTERING_STRATEGY,
    save: bool = True,
):
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
        parquet_table_save(
            query_output_dir,
            "create_base_entity_graph",
            dataset,
        )

    return dataset
