import df_ops
import os
from utils.save import parquet_table_load, parquet_table_save
import pandas as pd


def create_final_communities(
    base_entity_graph_output: pd.DataFrame,
    query_output_dir: str | None = None,
    save=True,
    try_load: bool = True,
):
    fn_name = "create_final_communities"
    if try_load and os.path.exists(
        os.path.join(query_output_dir, f"{fn_name}.parquet")
    ):
        return parquet_table_load(query_output_dir, fn_name)
    graph_nodes_output = df_ops.unpack_graph(
        base_entity_graph_output,
        **{
            "column": "clustered_graph",
            "type": "nodes",
        },
    )

    graph_edges_output = df_ops.unpack_graph(
        base_entity_graph_output,
        **{
            "column": "clustered_graph",
            "type": "edges",
        },
    )

    source_clusters_output = df_ops.join(
        table=graph_nodes_output,
        other=graph_edges_output,
        on=["label", "source"],
    )

    target_clusters_output = df_ops.join(
        table=graph_nodes_output,
        other=graph_edges_output,
        on=["label", "target"],
    )

    concatenated_clusters = df_ops.concat(
        table=source_clusters_output,
        others=[target_clusters_output],
    )

    combined_cluster = df_ops.filter_verb(
        concatenated_clusters,
        # level_1 is the left side of the join
        # level_2 is the right side of the join
        column="level_1",
        operator="equals",
        strategy="column",
        value="level_2",
    )

    cluster_relationships = df_ops.aggregate_override(
        combined_cluster,
        **{
            "groupby": [
                "cluster",
                "level_1",  # level_1 is the left side of the join
            ],
            "aggregations": [
                {
                    "column": "id_2",  # this is the id of the edge from the join steps above
                    "to": "relationship_ids",
                    "operation": "array_agg_distinct",
                },
                {
                    "column": "source_id_1",
                    "to": "text_unit_ids",
                    "operation": "array_agg_distinct",
                },
            ],
        },
    )

    all_clusters = df_ops.aggregate_override(
        graph_nodes_output,
        **{
            "groupby": ["cluster", "level"],
            "aggregations": [{"column": "cluster", "to": "id", "operation": "any"}],
        },
    )

    joined_dataset = df_ops.join(
        all_clusters, other=cluster_relationships, on=["id", "cluster"]
    )

    joined_dataset = df_ops.filter_verb(
        joined_dataset,
        strategy="column",
        column="level",
        operator="equals",
        value="level_1",
    )

    joined_dataset = df_ops.fill(
        joined_dataset,
        **{
            "to": "__temp",
            "value": "Community ",
        },
    )

    joined_dataset = df_ops.merge(
        joined_dataset,
        **{
            "columns": [
                "__temp",
                "id",
            ],
            "to": "title",
            "strategy": "concat",
            "preserveSource": True,
        },
    )

    joined_dataset = df_ops.copy(
        joined_dataset,
        **{
            "column": "id",
            "to": "raw_community",
        },
    )

    joined_dataset = df_ops.select(
        joined_dataset,
        **{
            "columns": [
                "id",
                "title",
                "level",
                "raw_community",
                "relationship_ids",
                "text_unit_ids",
            ],
        },
    )

    if save:
        parquet_table_save(query_output_dir, fn_name, joined_dataset)

    return joined_dataset
