from verbs.unpack import unpack_graph
from verbs.graphrag import *


def create_final_communities(base_entity_graph_output):

    graph_nodes_output = unpack_graph(
        base_entity_graph_output,
        **{
            "column": "clustered_graph",
            "type": "nodes",
        },
    )

    graph_edges_output = unpack_graph(
        base_entity_graph_output,
        **{
            "column": "clustered_graph",
            "type": "edges",
        },
    )

    source_clusters_output = join(
        table=graph_nodes_output,
        other=graph_edges_output,
        on=["label", "source"],
    )

    target_clusters_output = join(
        table=graph_nodes_output,
        other=graph_edges_output,
        on=["label", "target"],
    )

    concatenated_clusters = concat(
        table=source_clusters_output,
        others=[target_clusters_output],
    )

    combined_cluster = filter_verb(
        concatenated_clusters,
        # level_1 is the left side of the join
        # level_2 is the right side of the join
        column="level_1",
        operator="equals",
        strategy="column",
        value="level_2",
    )

    cluster_relationships = aggregate_override(
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

    all_clusters = aggregate_override(
        graph_nodes_output,
        **{
            "groupby": ["cluster", "level"],
            "aggregations": [{"column": "cluster", "to": "id", "operation": "any"}],
        },
    )

    joined_dataset = join(
        all_clusters, other=cluster_relationships, on=["id", "cluster"]
    )

    joined_dataset = filter_verb(
        joined_dataset,
        strategy="column",
        column="level",
        operator="equals",
        value="level_1",
    )

    joined_dataset = fill(
        joined_dataset,
        **{
            "to": "__temp",
            "value": "Community ",
        },
    )

    joined_dataset = merge(
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

    joined_dataset = copy(
        joined_dataset,
        **{
            "column": "id",
            "to": "raw_community",
        },
    )

    joined_dataset = select(
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

    return joined_dataset
