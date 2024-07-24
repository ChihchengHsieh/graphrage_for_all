from enum import Enum
from typing import Any, cast

import networkx as nx
import pandas as pd

import uuid
from random import Random, getrandbits
import logging

Communities = list[tuple[int, str, list[str]]]

def load_graph(graphml: str | nx.Graph) -> nx.Graph:
    """Load a graph from a graphml file or a networkx graph."""
    return nx.parse_graphml(graphml) if isinstance(graphml, str) else graphml


def gen_uuid(rd: Random | None = None):
    """Generate a random UUID v4."""
    return uuid.UUID(
        int=rd.getrandbits(128) if rd is not None else getrandbits(128), version=4
    ).hex


def cluster_graph(
    input: pd.DataFrame,
    # callbacks: VerbCallbacks,
    strategy: dict[str, Any],
    column: str,
    to: str,
    level_to: str | None = None,
    **_kwargs,
) -> pd.DataFrame:
    """
    Apply a hierarchical clustering algorithm to a graph. The graph is expected to be in graphml format. The verb outputs a new column containing the clustered graph, and a new column containing the level of the graph.

    ## Usage
    ```yaml
    verb: cluster_graph
    args:
        column: entity_graph # The name of the column containing the graph, should be a graphml graph
        to: clustered_graph # The name of the column to output the clustered graph to
        level_to: level # The name of the column to output the level to
        strategy: <strategy config> # See strategies section below
    ```

    ## Strategies
    The cluster graph verb uses a strategy to cluster the graph. The strategy is a json object which defines the strategy to use. The following strategies are available:

    ### leiden
    This strategy uses the leiden algorithm to cluster a graph. The strategy config is as follows:
    ```yaml
    strategy:
        type: leiden
        max_cluster_size: 10 # Optional, The max cluster size to use, default: 10
        use_lcc: true # Optional, if the largest connected component should be used with the leiden algorithm, default: true
        seed: 0xDEADBEEF # Optional, the seed to use for the leiden algorithm, default: 0xDEADBEEF
        levels: [0, 1] # Optional, the levels to output, default: all the levels detected

    ```
    """
    output_df = input
    results = output_df[column].apply(lambda graph: run_layout(strategy, graph))

    community_map_to = "communities"
    output_df[community_map_to] = results

    level_to = level_to or f"{to}_level"
    output_df[level_to] = output_df.apply(
        lambda x: list({level for level, _, _ in x[community_map_to]}), axis=1
    )
    output_df[to] = [None] * len(output_df)

    # Go through each of the rows
    graph_level_pairs_column: list[list[tuple[int, str]]] = []
    for _, row in output_df.iterrows():
        levels = row[level_to]
        graph_level_pairs: list[tuple[int, str]] = []

        # For each of the levels, get the graph and add it to the list
        for level in levels:
            graph = "\n".join(
                nx.generate_graphml(
                    apply_clustering(
                        row[column],
                        row[community_map_to],
                        level,
                    )
                )
            )
            graph_level_pairs.append((level, graph))
        graph_level_pairs_column.append(graph_level_pairs)
    output_df[to] = graph_level_pairs_column

    # explode the list of (level, graph) pairs into separate rows
    output_df = output_df.explode(to, ignore_index=True)

    # split the (level, graph) pairs into separate columns
    # TODO: There is probably a better way to do this
    output_df[[level_to, to]] = pd.DataFrame(
        output_df[to].tolist(), index=output_df.index
    )

    # clean up the community map
    output_df.drop(columns=[community_map_to], inplace=True)

    return output_df


# TODO: This should support str | nx.Graph as a graphml param
def apply_clustering(
    graphml: str, communities: Communities, level=0, seed=0xF001
) -> nx.Graph:
    """Apply clustering to a graphml string."""
    random = Random(seed)  # noqa S311
    graph = nx.parse_graphml(graphml)
    for community_level, community_id, nodes in communities:
        if level == community_level:
            for node in nodes:
                graph.nodes[node]["cluster"] = community_id
                graph.nodes[node]["level"] = level

    # add node degree
    for node_degree in graph.degree:
        graph.nodes[str(node_degree[0])]["degree"] = int(node_degree[1])

    # add node uuid and incremental record id (a human readable id used as reference in the final report)
    for index, node in enumerate(graph.nodes()):
        graph.nodes[node]["human_readable_id"] = index
        graph.nodes[node]["id"] = str(gen_uuid(random))

    # add ids to edges
    for index, edge in enumerate(graph.edges()):
        graph.edges[edge]["id"] = str(gen_uuid(random))
        graph.edges[edge]["human_readable_id"] = index
        graph.edges[edge]["level"] = level
    return graph


class GraphCommunityStrategyType(str, Enum):
    """GraphCommunityStrategyType class definition."""

    leiden = "leiden"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


def run_layout(
    strategy: dict[str, Any], graphml_or_graph: str | nx.Graph
) -> Communities:
    """Run layout method definition."""
    graph = load_graph(graphml_or_graph)
    if len(graph.nodes) == 0:
        logging.warning("Graph has no nodes")
        return []

    clusters: dict[int, dict[str, list[str]]] = {}
    strategy_type = strategy.get("type", GraphCommunityStrategyType.leiden)
    match strategy_type:
        case GraphCommunityStrategyType.leiden:
            from .leiden import run as run_leiden

            clusters = run_leiden(graph, strategy)
        case _:
            msg = f"Unknown clustering strategy {strategy_type}"
            raise ValueError(msg)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, nodes))
    return results
