import pandas as pd
import tiktoken
from . import schemas

_MISSING_DESCRIPTION = "No Description"


def prepare_community_reports_nodes(
    input: pd.DataFrame,
    to: str = schemas.NODE_DETAILS,
    id_column: str = schemas.NODE_ID,
    name_column: str = schemas.NODE_NAME,
    description_column: str = schemas.NODE_DESCRIPTION,
    degree_column: str = schemas.NODE_DEGREE,
    **_kwargs,
) -> pd.DataFrame:
    """Merge edge details into an object."""
    node_df = input
    node_df = node_df.fillna(value={description_column: _MISSING_DESCRIPTION})

    # merge values of four columns into a map column
    node_df[to] = node_df.apply(
        lambda x: {
            id_column: x[id_column],
            name_column: x[name_column],
            description_column: x[description_column],
            degree_column: x[degree_column],
        },
        axis=1,
    )
    return node_df


def prepare_community_reports_edges(
    input: pd.DataFrame,
    to: str = schemas.EDGE_DETAILS,
    id_column: str = schemas.EDGE_ID,
    source_column: str = schemas.EDGE_SOURCE,
    target_column: str = schemas.EDGE_TARGET,
    description_column: str = schemas.EDGE_DESCRIPTION,
    degree_column: str = schemas.EDGE_DEGREE,
    **_kwargs,
) -> pd.DataFrame:
    """Merge edge details into an object."""
    edge_df: pd.DataFrame = input.fillna(
        value={description_column: _MISSING_DESCRIPTION}
    )
    edge_df[to] = edge_df.apply(
        lambda x: {
            id_column: x[id_column],
            source_column: x[source_column],
            target_column: x[target_column],
            description_column: x[description_column],
            degree_column: x[degree_column],
        },
        axis=1,
    )
    return edge_df


def get_levels(df: pd.DataFrame, level_column: str = schemas.NODE_LEVEL) -> list[int]:
    """Get the levels of the communities."""
    result = sorted(df[level_column].fillna(-1).unique().tolist(), reverse=True)
    return [r for r in result if r != -1]


def prepare_community_reports(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    claim_df: pd.DataFrame | None = None,
    # callbacks: VerbCallbacks,
    max_tokens: int = 16_000,
    **_kwargs,
) -> pd.DataFrame:
    """Generate entities for each row, and optionally a graph of those entities."""
    # Prepare Community Reports
    # node_df = cast(pd.DataFrame, get_required_input_table(input, "nodes").table)
    # edge_df = cast(pd.DataFrame, get_required_input_table(input, "edges").table)
    # claim_df = get_named_input_table(input, "claims")
    if claim_df is not None:
        claim_df = claim_df

    levels = get_levels(node_df, schemas.NODE_LEVEL)
    dfs = []

    for level in levels:
        communities_at_level_df = _prepare_reports_at_level(
            node_df, edge_df, claim_df, level, max_tokens
        )
        dfs.append(communities_at_level_df)

    # build initial local context for all communities
    return pd.concat(dfs)


def filter_nodes_to_level(node_df: pd.DataFrame, level: int) -> pd.DataFrame:
    """Filter nodes to level."""
    return node_df[node_df[schemas.NODE_LEVEL] == level]


def filter_edges_to_nodes(edge_df: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """Filter edges to nodes."""
    return edge_df[
        edge_df[schemas.EDGE_SOURCE].isin(nodes)
        & edge_df[schemas.EDGE_TARGET].isin(nodes)
    ]


def filter_claims_to_nodes(claims_df: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """Filter edges to nodes."""
    return claims_df[claims_df[schemas.CLAIM_SUBJECT].isin(nodes)]


def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    return len(token_encoder.encode(text))  # type: ignore


def sort_context(
    local_context: list[dict],
    sub_community_reports: list[dict] | None = None,
    max_tokens: int | None = None,
    node_id_column: str = schemas.NODE_ID,
    node_name_column: str = schemas.NODE_NAME,
    node_details_column: str = schemas.NODE_DETAILS,
    edge_id_column: str = schemas.EDGE_ID,
    edge_details_column: str = schemas.EDGE_DETAILS,
    edge_degree_column: str = schemas.EDGE_DEGREE,
    edge_source_column: str = schemas.EDGE_SOURCE,
    edge_target_column: str = schemas.EDGE_TARGET,
    claim_id_column: str = schemas.CLAIM_ID,
    claim_details_column: str = schemas.CLAIM_DETAILS,
    community_id_column: str = schemas.COMMUNITY_ID,
) -> str:
    """Sort context by degree in descending order.

    If max tokens is provided, we will return the context string that fits within the token limit.
    """

    def _get_context_string(
        entities: list[dict],
        edges: list[dict],
        claims: list[dict],
        sub_community_reports: list[dict] | None = None,
    ) -> str:
        """Concatenate structured data into a context string."""
        contexts = []
        if sub_community_reports:
            sub_community_reports = [
                report
                for report in sub_community_reports
                if community_id_column in report
                and report[community_id_column]
                and str(report[community_id_column]).strip() != ""
            ]
            report_df = pd.DataFrame(sub_community_reports).drop_duplicates()
            if not report_df.empty:
                if report_df[community_id_column].dtype == float:
                    report_df[community_id_column] = report_df[
                        community_id_column
                    ].astype(int)
                report_string = (
                    f"----Reports-----\n{report_df.to_csv(index=False, sep=',')}"
                )
                contexts.append(report_string)

        entities = [
            entity
            for entity in entities
            if node_id_column in entity
            and entity[node_id_column]
            and str(entity[node_id_column]).strip() != ""
        ]
        entity_df = pd.DataFrame(entities).drop_duplicates()
        if not entity_df.empty:
            if entity_df[node_id_column].dtype == float:
                entity_df[node_id_column] = entity_df[node_id_column].astype(int)
            entity_string = (
                f"-----Entities-----\n{entity_df.to_csv(index=False, sep=',')}"
            )
            contexts.append(entity_string)

        if claims and len(claims) > 0:
            claims = [
                claim
                for claim in claims
                if claim_id_column in claim
                and claim[claim_id_column]
                and str(claim[claim_id_column]).strip() != ""
            ]
            claim_df = pd.DataFrame(claims).drop_duplicates()
            if not claim_df.empty:
                if claim_df[claim_id_column].dtype == float:
                    claim_df[claim_id_column] = claim_df[claim_id_column].astype(int)
                claim_string = (
                    f"-----Claims-----\n{claim_df.to_csv(index=False, sep=',')}"
                )
                contexts.append(claim_string)

        edges = [
            edge
            for edge in edges
            if edge_id_column in edge
            and edge[edge_id_column]
            and str(edge[edge_id_column]).strip() != ""
        ]
        edge_df = pd.DataFrame(edges).drop_duplicates()
        if not edge_df.empty:
            if edge_df[edge_id_column].dtype == float:
                edge_df[edge_id_column] = edge_df[edge_id_column].astype(int)
            edge_string = (
                f"-----Relationships-----\n{edge_df.to_csv(index=False, sep=',')}"
            )
            contexts.append(edge_string)

        return "\n\n".join(contexts)

    # sort node details by degree in descending order
    edges = []
    node_details = {}
    claim_details = {}

    for record in local_context:
        node_name = record[node_name_column]
        record_edges = record.get(edge_details_column, [])
        record_edges = [e for e in record_edges if not pd.isna(e)]
        record_node_details = record[node_details_column]
        record_claims = record.get(claim_details_column, [])
        record_claims = [c for c in record_claims if not pd.isna(c)]

        edges.extend(record_edges)
        node_details[node_name] = record_node_details
        claim_details[node_name] = record_claims

    edges = [edge for edge in edges if isinstance(edge, dict)]
    edges = sorted(edges, key=lambda x: x[edge_degree_column], reverse=True)

    sorted_edges = []
    sorted_nodes = []
    sorted_claims = []
    context_string = ""
    for edge in edges:
        source_details = node_details.get(edge[edge_source_column], {})
        target_details = node_details.get(edge[edge_target_column], {})
        sorted_nodes.extend([source_details, target_details])
        sorted_edges.append(edge)
        source_claims = claim_details.get(edge[edge_source_column], [])
        target_claims = claim_details.get(edge[edge_target_column], [])
        sorted_claims.extend(source_claims if source_claims else [])
        sorted_claims.extend(target_claims if source_claims else [])
        if max_tokens:
            new_context_string = _get_context_string(
                sorted_nodes, sorted_edges, sorted_claims, sub_community_reports
            )
            if num_tokens(context_string) > max_tokens:
                break
            context_string = new_context_string

    if context_string == "":
        return _get_context_string(
            sorted_nodes, sorted_edges, sorted_claims, sub_community_reports
        )

    return context_string


def set_context_size(df: pd.DataFrame) -> None:
    """Measure the number of tokens in the context."""
    df[schemas.CONTEXT_SIZE] = df[schemas.CONTEXT_STRING].apply(lambda x: num_tokens(x))


def set_context_exceeds_flag(df: pd.DataFrame, max_tokens: int) -> None:
    """Set a flag to indicate if the context exceeds the limit."""
    df[schemas.CONTEXT_EXCEED_FLAG] = df[schemas.CONTEXT_SIZE].apply(
        lambda x: x > max_tokens
    )


def _prepare_reports_at_level(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    claim_df: pd.DataFrame | None,
    level: int,
    max_tokens: int = 16_000,
    community_id_column: str = schemas.COMMUNITY_ID,
    node_id_column: str = schemas.NODE_ID,
    node_name_column: str = schemas.NODE_NAME,
    node_details_column: str = schemas.NODE_DETAILS,
    node_level_column: str = schemas.NODE_LEVEL,
    node_degree_column: str = schemas.NODE_DEGREE,
    node_community_column: str = schemas.NODE_COMMUNITY,
    edge_id_column: str = schemas.EDGE_ID,
    edge_source_column: str = schemas.EDGE_SOURCE,
    edge_target_column: str = schemas.EDGE_TARGET,
    edge_degree_column: str = schemas.EDGE_DEGREE,
    edge_details_column: str = schemas.EDGE_DETAILS,
    claim_id_column: str = schemas.CLAIM_ID,
    claim_subject_column: str = schemas.CLAIM_SUBJECT,
    claim_details_column: str = schemas.CLAIM_DETAILS,
):
    def get_edge_details(node_df: pd.DataFrame, edge_df: pd.DataFrame, name_col: str):
        return node_df.merge(
            edge_df[[name_col, schemas.EDGE_DETAILS]].rename(
                columns={name_col: schemas.NODE_NAME}
            ),
            on=schemas.NODE_NAME,
            how="left",
        )

    level_node_df = filter_nodes_to_level(node_df, level)
    # log.info("Number of nodes at level=%s => %s", level, len(level_node_df))
    nodes = level_node_df[node_name_column].tolist()

    # Filter edges & claims to those containing the target nodes
    level_edge_df = filter_edges_to_nodes(edge_df, nodes)
    level_claim_df = (
        filter_claims_to_nodes(claim_df, nodes) if claim_df is not None else None
    )

    # concat all edge details per node
    merged_node_df = pd.concat(
        [
            get_edge_details(level_node_df, level_edge_df, edge_source_column),
            get_edge_details(level_node_df, level_edge_df, edge_target_column),
        ],
        axis=0,
    )
    merged_node_df = (
        merged_node_df.groupby(
            [
                node_name_column,
                node_community_column,
                node_degree_column,
                node_level_column,
            ]
        )
        .agg({node_details_column: "first", edge_details_column: list})
        .reset_index()
    )

    # concat claim details per node
    if level_claim_df is not None:
        merged_node_df = merged_node_df.merge(
            level_claim_df[[claim_subject_column, claim_details_column]].rename(
                columns={claim_subject_column: node_name_column}
            ),
            on=node_name_column,
            how="left",
        )
    merged_node_df = (
        merged_node_df.groupby(
            [
                node_name_column,
                node_community_column,
                node_level_column,
                node_degree_column,
            ]
        )
        .agg(
            {
                node_details_column: "first",
                edge_details_column: "first",
                **({claim_details_column: list} if level_claim_df is not None else {}),
            }
        )
        .reset_index()
    )

    # concat all node details, including name, degree, node_details, edge_details, and claim_details
    merged_node_df[schemas.ALL_CONTEXT] = merged_node_df.apply(
        lambda x: {
            node_name_column: x[node_name_column],
            node_degree_column: x[node_degree_column],
            node_details_column: x[node_details_column],
            edge_details_column: x[edge_details_column],
            claim_details_column: (
                x[claim_details_column] if level_claim_df is not None else []
            ),
        },
        axis=1,
    )

    # group all node details by community
    community_df = (
        merged_node_df.groupby(node_community_column)
        .agg({schemas.ALL_CONTEXT: list})
        .reset_index()
    )
    community_df[schemas.CONTEXT_STRING] = community_df[schemas.ALL_CONTEXT].apply(
        lambda x: sort_context(
            x,
            node_id_column=node_id_column,
            node_name_column=node_name_column,
            node_details_column=node_details_column,
            edge_id_column=edge_id_column,
            edge_details_column=edge_details_column,
            edge_degree_column=edge_degree_column,
            edge_source_column=edge_source_column,
            edge_target_column=edge_target_column,
            claim_id_column=claim_id_column,
            claim_details_column=claim_details_column,
            community_id_column=community_id_column,
        )
    )
    set_context_size(community_df)
    set_context_exceeds_flag(community_df, max_tokens)

    community_df[schemas.COMMUNITY_LEVEL] = level
    return community_df
