import pandas as pd
from typing import Any, cast
from enum import Enum
from dataclasses import dataclass
import networkx as nx
from utils.graph import load_graph


DEFAULT_CONCAT_SEPARATOR = ","


class NumericOperation(str, Enum):
    """Numeric Operation class definition."""

    Sum = "sum"
    Average = "average"
    Max = "max"
    Min = "min"
    Multiply = "multiply"
    Replace = "replace"
    Skip = "skip"


class StringOperation(str, Enum):
    """String Operation class definition."""

    Concat = "concat"
    Replace = "replace"
    Skip = "skip"


class BasicMergeOperation(str, Enum):
    """Basic Merge Operation class definition."""

    Replace = "replace"
    Skip = "skip"


DEFAULT_NODE_OPERATIONS = {
    "*": {
        "operation": BasicMergeOperation.Replace,
    }
}

DEFAULT_EDGE_OPERATIONS = {
    "*": {
        "operation": BasicMergeOperation.Replace,
    },
    "weight": "sum",
}


@dataclass
class DetailedAttributeMergeOperation:
    """Detailed attribute merge operation class definition."""

    operation: StringOperation | NumericOperation

    # concat
    separator: str | None = None
    delimiter: str | None = None
    distinct: bool = False


def _get_detailed_attribute_merge_operation(
    value: str | dict[str, Any],
) -> DetailedAttributeMergeOperation:
    """Normalize the AttributeMergeOperation into a DetailedAttributeMergeOperation."""
    if isinstance(value, str):
        return DetailedAttributeMergeOperation(operation=value)
    return DetailedAttributeMergeOperation(**value)


def merge_edges(
    target_graph: nx.Graph,
    subgraph: nx.Graph,
    edge_ops: dict[str, DetailedAttributeMergeOperation],
):
    """Merge edges from subgraph into target using the operations defined in edge_ops."""
    for source, target, edge_data in subgraph.edges(data=True):  # type: ignore
        if not target_graph.has_edge(source, target):
            target_graph.add_edge(source, target, **(edge_data or {}))
        else:
            merge_attributes(target_graph.edges[(source, target)], edge_data, edge_ops)


def merge_nodes(
    target: nx.Graph,
    subgraph: nx.Graph,
    node_ops: dict[str, DetailedAttributeMergeOperation],
):
    """Merge nodes from subgraph into target using the operations defined in node_ops."""
    for node in subgraph.nodes:
        if node not in target.nodes:
            target.add_node(node, **(subgraph.nodes[node] or {}))
        else:
            merge_attributes(target.nodes[node], subgraph.nodes[node], node_ops)


def merge_graphs(
    input: pd.DataFrame,
    # callbacks: VerbCallbacks,
    column: str,
    to: str,
    nodes: dict[str, Any] = DEFAULT_NODE_OPERATIONS,
    edges: dict[str, Any] = DEFAULT_EDGE_OPERATIONS,
    **_kwargs,
) -> pd.DataFrame:

    input_df = input
    output = pd.DataFrame()

    node_ops = {
        attrib: _get_detailed_attribute_merge_operation(value)
        for attrib, value in nodes.items()
    }
    edge_ops = {
        attrib: _get_detailed_attribute_merge_operation(value)
        for attrib, value in edges.items()
    }

    mega_graph = nx.Graph()
    for graphml in input_df[column]:
        graph = load_graph(cast(str | nx.Graph, graphml))
        merge_nodes(mega_graph, graph, node_ops)
        merge_edges(mega_graph, graph, edge_ops)

    output[to] = ["\n".join(nx.generate_graphml(mega_graph))]

    return output


def apply_merge_operation(
    target_item: dict[str, Any] | None,
    source_item: dict[str, Any] | None,
    attrib: str,
    op: DetailedAttributeMergeOperation,
):
    """Apply the merge operation to the attribute."""
    source_item = source_item or {}
    target_item = target_item or {}

    if (
        op.operation == BasicMergeOperation.Replace
        or op.operation == StringOperation.Replace
    ):
        target_item[attrib] = source_item.get(attrib, None) or ""
    elif (
        op.operation == BasicMergeOperation.Skip or op.operation == StringOperation.Skip
    ):
        target_item[attrib] = target_item.get(attrib, None) or ""
    elif op.operation == StringOperation.Concat:
        separator = op.separator or DEFAULT_CONCAT_SEPARATOR
        target_attrib = target_item.get(attrib, "") or ""
        source_attrib = source_item.get(attrib, "") or ""
        target_item[attrib] = f"{target_attrib}{separator}{source_attrib}"
        if op.distinct:
            # TODO: Slow
            target_item[attrib] = separator.join(
                sorted(set(target_item[attrib].split(separator)))
            )

    # We're assuming that the attribute is numeric
    elif op.operation == NumericOperation.Sum:
        target_item[attrib] = (target_item.get(attrib, 0) or 0) + (
            source_item.get(attrib, 0) or 0
        )
    elif op.operation == NumericOperation.Average:
        target_item[attrib] = (
            (target_item.get(attrib, 0) or 0) + (source_item.get(attrib, 0) or 0)
        ) / 2
    elif op.operation == NumericOperation.Max:
        target_item[attrib] = max(
            (target_item.get(attrib, 0) or 0), (source_item.get(attrib, 0) or 0)
        )
    elif op.operation == NumericOperation.Min:
        target_item[attrib] = min(
            (target_item.get(attrib, 0) or 0), (source_item.get(attrib, 0) or 0)
        )
    elif op.operation == NumericOperation.Multiply:
        target_item[attrib] = (target_item.get(attrib, 1) or 1) * (
            source_item.get(attrib, 1) or 1
        )
    else:
        msg = f"Invalid operation {op.operation}"
        raise ValueError(msg)


def merge_attributes(
    target_item: dict[str, Any] | None,
    source_item: dict[str, Any] | None,
    ops: dict[str, DetailedAttributeMergeOperation],
):
    """Merge attributes from source_item into target_item using the operations defined in ops."""
    source_item = source_item or {}
    target_item = target_item or {}
    for op_attrib, op in ops.items():
        if op_attrib == "*":
            for attrib in source_item:
                # If there is a specific handler for this attribute, use it
                # i.e. * provides a default, but you can override it
                if attrib not in ops:
                    apply_merge_operation(target_item, source_item, attrib, op)
        else:
            if op_attrib in source_item or op_attrib in target_item:
                apply_merge_operation(target_item, source_item, op_attrib, op)
