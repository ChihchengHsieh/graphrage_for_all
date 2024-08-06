import pandas as pd
from typing import cast


def _degree_colname(column: str) -> str:
    return f"{column}_degree"


def _get_node_degree_table(
    nodes: pd.DataFrame, node_name_column: str, node_degree_column: str
) -> pd.DataFrame:
    return cast(pd.DataFrame, nodes[[node_name_column, node_degree_column]])


def compute_edge_combined_degree(
    input: pd.DataFrame,
    nodes: pd.DataFrame,
    to: str = "rank",
    node_name_column: str = "title",
    node_degree_column: str = "degree",
    edge_source_column: str = "source",
    edge_target_column: str = "target",
    **_kwargs,
) -> pd.DataFrame:
    """
    Compute the combined degree for each edge in a graph.

    Inputs Tables:
    - input: The edge table
    - nodes: The nodes table.

    Args:
    - to: The name of the column to output the combined degree to. Default="rank"
    """
    edge_df: pd.DataFrame = input
    if to in edge_df.columns:
        return edge_df
    node_degree_df = _get_node_degree_table(nodes, node_name_column, node_degree_column)

    def join_to_degree(df: pd.DataFrame, column: str) -> pd.DataFrame:
        degree_column = _degree_colname(column)
        result = df.merge(
            node_degree_df.rename(
                columns={node_name_column: column, node_degree_column: degree_column}
            ),
            on=column,
            how="left",
        )
        result[degree_column] = result[degree_column].fillna(0)
        return result

    edge_df = join_to_degree(edge_df, edge_source_column)
    edge_df = join_to_degree(edge_df, edge_target_column)
    edge_df[to] = (
        edge_df[_degree_colname(edge_source_column)]
        + edge_df[_degree_colname(edge_target_column)]
    )

    return edge_df
