from verbs.unpack import unpack_graph
from verbs.layout import layout_graph
from copy import deepcopy
from verbs.graphrag import *

layout_graph_config = {"strategy": {"type": "zero"}}


def create_final_nodes(base_entity_graph_output):
    laid_out_entity_graph = layout_graph(
        deepcopy(base_entity_graph_output),
        **{
            "embeddings_column": "embeddings",
            "graph_column": "clustered_graph",
            "to": "node_positions",
            "graph_to": "positioned_graph",
            **layout_graph_config,
        },
    )

    nodes_without_positions = unpack_graph(
        deepcopy(laid_out_entity_graph),
        **{"column": "positioned_graph", "type": "nodes"},
    )

    nodes_without_positions = drop(
        nodes_without_positions,
        **{"columns": ["x", "y"]},
    )

    compute_top_level_node_positions = unpack_graph(
        deepcopy(laid_out_entity_graph),
        **{"column": "positioned_graph", "type": "nodes"},
    )

    compute_top_level_node_positions = filter_verb(
        compute_top_level_node_positions,
        column="level",
        value=0,
        strategy="value",
        operator="equals",
    )

    compute_top_level_node_positions = select(
        compute_top_level_node_positions,
        **{"columns": ["id", "x", "y"]},
    )

    compute_top_level_node_positions = rename(
        compute_top_level_node_positions,
        columns={
            "id": "top_level_node_id",
        },
    )

    compute_top_level_node_positions = convert(
        compute_top_level_node_positions,
        **{
            "column": "top_level_node_id",
            "to": "top_level_node_id",
            "type": "string",
        },
    )

    dataset = join(
        nodes_without_positions,
        compute_top_level_node_positions,
        on=["id", "top_level_node_id"],
    )

    dataset = rename(dataset, **{"columns": {"label": "title", "cluster": "community"}})

    return dataset
