import os
from graphrag_for_all import df_ops
import pandas as pd

from graphrag_for_all.utils.save import parquet_table_load, parquet_table_save


def create_final_relationships(
    base_entity_graph_output: pd.DataFrame,
    final_nodes_output: pd.DataFrame,
    query_output_dir: str,
    save: bool = True,
    try_load: bool = True,
):
    fn_name = "create_final_relationships"
    if try_load and os.path.exists(
        os.path.join(query_output_dir, f"{fn_name}.parquet")
    ):
        return parquet_table_load(query_output_dir, fn_name)

    dataset = df_ops.unpack_graph(
        base_entity_graph_output,
        **{
            "column": "clustered_graph",
            "type": "edges",
        },
    )

    dataset = df_ops.rename(
        dataset,
        **{"columns": {"source_id": "text_unit_ids"}},
    )

    dataset = df_ops.filter_verb(
        dataset,
        column="level",
        strategy="value",
        operator="equals",
        value=0,
    )

    pruned_edges = df_ops.drop(
        dataset,
        columns=["level"],
    )

    filtered_nodes = df_ops.filter_verb(
        final_nodes_output,
        column="level",
        strategy="value",
        operator="equals",
        value=0,
    )

    dataset = df_ops.compute_edge_combined_degree(
        input=pruned_edges,
        nodes=filtered_nodes,
        **{"to": "rank"},
    )

    dataset = df_ops.convert(
        dataset,
        **{
            "column": "human_readable_id",
            "type": "string",
            "to": "human_readable_id",
        },
    )

    dataset = df_ops.convert(
        dataset,
        **{
            "column": "text_unit_ids",
            "type": "array",
            "to": "text_unit_ids",
        },
    )

    if save:
        parquet_table_save(query_output_dir, fn_name, dataset)

    return dataset
