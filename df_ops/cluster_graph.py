import networkx as nx
import pandas as pd
import html

from typing import Any
from utils.graph import load_graph
from utils.hash import gen_uuid
from enum import Enum
from random import Random

from graspologic.partition import hierarchical_leiden
from graspologic.utils import largest_connected_component


def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    """Normalize node names."""
    node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
    return nx.relabel_nodes(graph, node_mapping)


def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """Ensure an undirected graph with the same relationships will always be read the same way."""
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    sorted_nodes = graph.nodes(data=True)
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

    fixed_graph.add_nodes_from(sorted_nodes)
    edges = list(graph.edges(data=True))

    # If the graph is undirected, we create the edges in a stable way, so we get the same results
    # for example:
    # A -> B
    # in graph theory is the same as
    # B -> A
    # in an undirected graph
    # however, this can lead to downstream issues because sometimes
    # consumers read graph.nodes() which ends up being [A, B] and sometimes it's [B, A]
    # but they base some of their logic on the order of the nodes, so the order ends up being important
    # so we sort the nodes in the edge in a stable way, so that we always get the same order
    if not graph.is_directed():

        def _sort_source_target(edge):
            source, target, edge_data = edge
            if source > target:
                temp = source
                source = target
                target = temp
            return source, target, edge_data

        edges = [_sort_source_target(edge) for edge in edges]

    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"

    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

    fixed_graph.add_edges_from(edges)
    return fixed_graph


def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component of the graph, with nodes and edges sorted in a stable way."""
    graph = graph.copy()
    graph = largest_connected_component(graph)
    graph = normalize_node_names(graph)
    return _stabilize_graph(graph)


def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed=0xDEADBEEF,
) -> dict[int, dict[str, int]]:
    """Return Leiden root communities."""
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

    return results


def run_leiden(
    graph: nx.Graph, args: dict[str, Any]
) -> dict[int, dict[str, list[str]]]:
    """Run method definition."""
    max_cluster_size = args.get("max_cluster_size", 10)
    use_lcc = args.get("use_lcc", True)
    if args.get("verbose", False):
        print(
            "Running leiden with max_cluster_size=%s, lcc=%s", max_cluster_size, use_lcc
        )

    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 0xDEADBEEF),
    )
    levels = args.get("levels")

    # If they don't pass in levels, use them all
    if levels is None:
        levels = sorted(node_id_to_community_map.keys())

    results_by_level: dict[int, dict[str, list[str]]] = {}
    for level in levels:
        result = {}
        results_by_level[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = str(raw_community_id)
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)
    return results_by_level


Communities = list[tuple[int, str, list[str]]]


class GraphCommunityStrategyType(str, Enum):
    """GraphCommunityStrategyType class definition."""

    leiden = "leiden"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


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


def run_layout(
    strategy: dict[str, Any], graphml_or_graph: str | nx.Graph
) -> Communities:
    """Run layout method definition."""
    graph = load_graph(graphml_or_graph)
    if len(graph.nodes) == 0:
        print("Graph has no nodes")
        return []

    clusters: dict[int, dict[str, list[str]]] = {}
    strategy_type = strategy.get("type", GraphCommunityStrategyType.leiden)
    match strategy_type:
        case GraphCommunityStrategyType.leiden:
            clusters = run_leiden(graph, strategy)
        case _:
            msg = f"Unknown clustering strategy {strategy_type}"
            raise ValueError(msg)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, nodes))
    return results


def cluster_graph(
    input: pd.DataFrame,
    # callbacks: VerbCallbacks,
    strategy: dict[str, Any],
    column: str,
    to: str,
    level_to: str | None = None,
    **_kwargs,
) -> pd.DataFrame:
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
