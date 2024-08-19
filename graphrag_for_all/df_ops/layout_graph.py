from enum import Enum
from typing import Any, cast
import networkx as nx
import pandas as pd
from dataclasses import dataclass
from graphrag_for_all.utils.graph import load_graph
import umap
import numpy as np


class LayoutGraphStrategyType(str, Enum):
    """LayoutGraphStrategyType class definition."""

    umap = "umap"
    zero = "zero"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


@dataclass
class NodePosition:
    """Node position class definition."""

    label: str
    cluster: str
    size: float

    x: float
    y: float
    z: float | None = None

    def to_pandas(self) -> tuple[str, float, float, str, float]:
        """To pandas method definition."""
        return self.label, self.x, self.y, self.cluster, self.size


GraphLayout = list[NodePosition]
NodeEmbeddings = dict[str, list[float]]


def layout_graph(
    input: pd.DataFrame,
    # callbacks: VerbCallbacks,
    strategy: dict[str, Any],
    embeddings_column: str,
    graph_column: str,
    to: str,
    graph_to: str | None = None,
    **_kwargs: dict,
) -> pd.DataFrame:

    output_df = input

    # num_items = len(output_df)
    strategy_type = strategy.get("type", LayoutGraphStrategyType.umap)
    strategy_args = {**strategy}

    has_embeddings = embeddings_column in output_df.columns

    layouts = output_df.apply(
        lambda row: _run_layout(
            strategy_type,
            row[graph_column],
            row[embeddings_column] if has_embeddings else {},
            strategy_args,
        ),
        axis=1,
    )
    output_df[to] = layouts.apply(lambda layout: [pos.to_pandas() for pos in layout])
    if graph_to is not None:
        output_df[graph_to] = output_df.apply(
            lambda row: _apply_layout_to_graph(
                row[graph_column], cast(GraphLayout, layouts[row.name])
            ),
            axis=1,
        )
    return output_df


def compute_umap_positions(
    embedding_vectors: np.ndarray,
    node_labels: list[str],
    node_categories: list[int] | None = None,
    node_sizes: list[int] | None = None,
    min_dist: float = 0.75,
    n_neighbors: int = 25,
    spread: int = 1,
    metric: str = "euclidean",
    n_components: int = 2,
    random_state: int = 86,
) -> list[NodePosition]:
    """Project embedding vectors down to 2D/3D using UMAP."""
    embedding_positions = umap.UMAP(
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        spread=spread,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    ).fit_transform(embedding_vectors)

    embedding_position_data: list[NodePosition] = []
    for index, node_name in enumerate(node_labels):
        node_points = embedding_positions[index]  # type: ignore
        node_category = 1 if node_categories is None else node_categories[index]
        node_size = 1 if node_sizes is None else node_sizes[index]

        if len(node_points) == 2:
            embedding_position_data.append(
                NodePosition(
                    label=str(node_name),
                    x=float(node_points[0]),
                    y=float(node_points[1]),
                    cluster=str(int(node_category)),
                    size=int(node_size),
                )
            )
        else:
            embedding_position_data.append(
                NodePosition(
                    label=str(node_name),
                    x=float(node_points[0]),
                    y=float(node_points[1]),
                    z=float(node_points[2]),
                    cluster=str(int(node_category)),
                    size=int(node_size),
                )
            )
    return embedding_position_data


def run_umap(
    graph: nx.Graph,
    embeddings: NodeEmbeddings,
    args: dict[str, Any],
) -> GraphLayout:
    """Run method definition."""
    node_clusters = []
    node_sizes = []

    embeddings = _filter_raw_embeddings(embeddings)
    nodes = list(embeddings.keys())
    embedding_vectors = [embeddings[node_id] for node_id in nodes]

    for node_id in nodes:
        node = graph.nodes[node_id]
        cluster = node.get("cluster", node.get("community", -1))
        node_clusters.append(cluster)
        size = node.get("degree", node.get("size", 0))
        node_sizes.append(size)

    additional_args = {}
    if len(node_clusters) > 0:
        additional_args["node_categories"] = node_clusters
    if len(node_sizes) > 0:
        additional_args["node_sizes"] = node_sizes

    try:
        return compute_umap_positions(
            embedding_vectors=np.array(embedding_vectors),
            node_labels=nodes,
            **additional_args,
            min_dist=args.get("min_dist", 0.75),
            n_neighbors=args.get("n_neighbors", 5),
        )
    except Exception as e:
        print("Error running UMAP")
        # Umap may fail due to input sparseness or memory pressure.
        # For now, in these cases, we'll just return a layout with all nodes at (0, 0)
        result = []
        for i in range(len(nodes)):
            cluster = node_clusters[i] if len(node_clusters) > 0 else 1
            result.append(
                NodePosition(x=0, y=0, label=nodes[i], size=0, cluster=str(cluster))
            )
        return result


def _filter_raw_embeddings(embeddings: NodeEmbeddings) -> NodeEmbeddings:
    return {
        node_id: embedding
        for node_id, embedding in embeddings.items()
        if embedding is not None
    }


def get_zero_positions(
    node_labels: list[str],
    node_categories: list[int] | None = None,
    node_sizes: list[int] | None = None,
    three_d: bool | None = False,
) -> list[NodePosition]:
    """Project embedding vectors down to 2D/3D using UMAP."""
    embedding_position_data: list[NodePosition] = []
    for index, node_name in enumerate(node_labels):
        node_category = 1 if node_categories is None else node_categories[index]
        node_size = 1 if node_sizes is None else node_sizes[index]

        if not three_d:
            embedding_position_data.append(
                NodePosition(
                    label=str(node_name),
                    x=0,
                    y=0,
                    cluster=str(int(node_category)),
                    size=int(node_size),
                )
            )
        else:
            embedding_position_data.append(
                NodePosition(
                    label=str(node_name),
                    x=0,
                    y=0,
                    z=0,
                    cluster=str(int(node_category)),
                    size=int(node_size),
                )
            )
    return embedding_position_data


def run_zero(
    graph: nx.Graph,
    _args: dict[str, Any],
) -> GraphLayout:
    """Run method definition."""
    node_clusters = []
    node_sizes = []

    nodes = list(graph.nodes)

    for node_id in nodes:
        node = graph.nodes[node_id]
        cluster = node.get("cluster", node.get("community", -1))
        node_clusters.append(cluster)
        size = node.get("degree", node.get("size", 0))
        node_sizes.append(size)

    additional_args = {}
    if len(node_clusters) > 0:
        additional_args["node_categories"] = node_clusters
    if len(node_sizes) > 0:
        additional_args["node_sizes"] = node_sizes

    try:
        return get_zero_positions(node_labels=nodes, **additional_args)
    except Exception as e:
        print("Error running zero-position")
        # Umap may fail due to input sparseness or memory pressure.
        # For now, in these cases, we'll just return a layout with all nodes at (0, 0)
        result = []
        for i in range(len(nodes)):
            cluster = node_clusters[i] if len(node_clusters) > 0 else 1
            result.append(
                NodePosition(x=0, y=0, label=nodes[i], size=0, cluster=str(cluster))
            )
        return result


def _run_layout(
    strategy: LayoutGraphStrategyType,
    graphml_or_graph: str | nx.Graph,
    embeddings: NodeEmbeddings,
    args: dict[str, Any],
    # reporter: VerbCallbacks,
) -> GraphLayout:
    graph = load_graph(graphml_or_graph)
    match strategy:
        case LayoutGraphStrategyType.umap:
            return run_umap(
                graph,
                embeddings,
                args,
            )
        case LayoutGraphStrategyType.zero:
            return run_zero(
                graph,
                args,
            )
        case _:
            msg = f"Unknown strategy {strategy}"
            raise ValueError(msg)


def _apply_layout_to_graph(
    graphml_or_graph: str | nx.Graph, layout: GraphLayout
) -> str:
    graph = load_graph(graphml_or_graph)
    for node_position in layout:
        if node_position.label in graph.nodes:
            graph.nodes[node_position.label]["x"] = node_position.x
            graph.nodes[node_position.label]["y"] = node_position.y
            graph.nodes[node_position.label]["size"] = node_position.size
    return "\n".join(nx.generate_graphml(graph))
