from graphrag_for_all import df_ops
import pandas as pd
from copy import deepcopy

import os
from graphrag_for_all.utils.save import parquet_table_load, parquet_table_save


layout_graph_config = {"strategy": {"type": "zero"}}


def create_final_nodes(
    base_entity_graph_output: pd.DataFrame,
    query_output_dir: str,
    save: bool = True,
    try_load: bool = True,
):

    fn_name = "create_final_nodes"
    if try_load and os.path.exists(
        os.path.join(query_output_dir, f"{fn_name}.parquet")
    ):
        return parquet_table_load(query_output_dir, fn_name)
    laid_out_entity_graph = df_ops.layout_graph(
        deepcopy(base_entity_graph_output),
        **{
            "embeddings_column": "embeddings",
            "graph_column": "clustered_graph",
            "to": "node_positions",
            "graph_to": "positioned_graph",
            **layout_graph_config,
        },
    )

    nodes_without_positions = df_ops.unpack_graph(
        deepcopy(laid_out_entity_graph),
        **{"column": "positioned_graph", "type": "nodes"},
    )

    nodes_without_positions = df_ops.drop(
        nodes_without_positions,
        **{"columns": ["x", "y"]},
    )

    compute_top_level_node_positions = df_ops.unpack_graph(
        deepcopy(laid_out_entity_graph),
        **{"column": "positioned_graph", "type": "nodes"},
    )

    compute_top_level_node_positions = df_ops.filter_verb(
        compute_top_level_node_positions,
        column="level",
        value=0,
        strategy="value",
        operator="equals",
    )

    compute_top_level_node_positions = df_ops.select(
        compute_top_level_node_positions,
        **{"columns": ["id", "x", "y"]},
    )

    compute_top_level_node_positions = df_ops.rename(
        compute_top_level_node_positions,
        columns={
            "id": "top_level_node_id",
        },
    )

    compute_top_level_node_positions = df_ops.convert(
        compute_top_level_node_positions,
        **{
            "column": "top_level_node_id",
            "to": "top_level_node_id",
            "type": "string",
        },
    )

    dataset = df_ops.join(
        nodes_without_positions,
        compute_top_level_node_positions,
        on=["id", "top_level_node_id"],
    )

    dataset = df_ops.rename(
        dataset, **{"columns": {"label": "title", "cluster": "community"}}
    )

    if save:
        parquet_table_save(query_output_dir, fn_name, dataset)

    return dataset
