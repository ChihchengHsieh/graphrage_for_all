from typing import Callable, Any
import pandas as pd
from . import schemas

from typing_extensions import TypedDict
from .send_to import send_to_open_ai


def get_levels(df: pd.DataFrame, level_column: str = schemas.NODE_LEVEL) -> list[int]:
    """Get the levels of the communities."""
    result = sorted(df[level_column].fillna(-1).unique().tolist(), reverse=True)
    return [r for r in result if r != -1]


class Finding(TypedDict):
    """Finding class definition."""

    summary: str
    explanation: str


class CommunityReport(TypedDict):
    """Community report class definition."""

    community: str | int
    title: str
    summary: str
    full_content: str
    full_content_json: str
    rank: float
    level: int
    rank_explanation: str
    findings: list[Finding]


StrategyConfig = dict[str, Any]

from .community_report_extractor import CommunityReportsExtractor
import json


def _parse_rank(report: dict) -> float:
    rank = report.get("rating", -1)
    try:
        return float(rank)
    except ValueError:
        print("Error parsing rank: %s defaulting to -1", rank)
        return -1


def _run_extractor(
    send_to: Callable,
    community: str | int,
    input: str,
    level: int,
    args: StrategyConfig,
    # reporter: VerbCallbacks,
) -> CommunityReport | None:
    # RateLimiter

    extractor = CommunityReportsExtractor(
        send_to,
        extraction_prompt=args.get("extraction_prompt", None),
        max_report_length=args.get("max_report_length", None),
        # on_error=lambda e, stack, _data: reporter.error(
        #     "Community Report Extraction Error", e, stack
        # ),
    )

    try:
        results = extractor({"input_text": input})
        report = results.structured_output
        if report is None or len(report.keys()) == 0:
            print("No report found for community: %s", community)
            return None

        return CommunityReport(
            community=community,
            full_content=results.output,
            level=level,
            rank=_parse_rank(report),
            title=report.get("title", f"Community Report: {community}"),
            rank_explanation=report.get("rating_explanation", ""),
            summary=report.get("summary", ""),
            findings=report.get("findings", []),
            full_content_json=json.dumps(report, indent=4),
        )
    except Exception as e:
        print("Error processing community: %s", community)
        return None


def where_column_equals(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    """Return a filtered DataFrame where a column equals a value."""
    return df[df[column] == value]


def _at_level(level: int, df: pd.DataFrame) -> pd.DataFrame:
    """Return records at the given level."""
    return where_column_equals(df, schemas.COMMUNITY_LEVEL, level)


def _within_context(df: pd.DataFrame) -> pd.DataFrame:
    """Return records where the context is within the limit."""
    return where_column_equals(df, schemas.CONTEXT_EXCEED_FLAG, 0)


def _exceeding_context(df: pd.DataFrame) -> pd.DataFrame:
    """Return records where the context exceeds the limit."""
    return where_column_equals(df, schemas.CONTEXT_EXCEED_FLAG, 1)


def transform_series(series: pd.Series, fn: Callable[[Any], Any]) -> pd.Series:
    """Apply a transformation function to a series."""
    return series.apply(fn)


import tiktoken


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


def _sort_and_trim_context(df: pd.DataFrame, max_tokens: int) -> pd.Series:
    """Sort and trim context to fit the limit."""
    series = df[schemas.ALL_CONTEXT]
    return transform_series(series, lambda x: sort_context(x, max_tokens=max_tokens))


def set_context_size(df: pd.DataFrame) -> None:
    """Measure the number of tokens in the context."""
    df[schemas.CONTEXT_SIZE] = df[schemas.CONTEXT_STRING].apply(lambda x: num_tokens(x))


def union(*frames: pd.DataFrame) -> pd.DataFrame:
    """Perform a union operation on the given set of dataframes."""
    return pd.concat(list(frames))


def antijoin(df: pd.DataFrame, exclude: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return an anti-joined dataframe.

    Arguments:
    * df: The DataFrame to apply the exclusion to
    * exclude: The DataFrame containing rows to remove.
    * column: The join-on column.
    """
    result = df.merge(
        exclude[[column]],
        on=column,
        how="outer",
        indicator=True,
    )
    if "_merge" in result.columns:
        result = result[result["_merge"] == "left_only"].drop("_merge", axis=1)
    return result


def _antijoin_reports(df: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Return records in df that are not in reports."""
    return antijoin(df, reports, schemas.NODE_COMMUNITY)


def drop_columns(df: pd.DataFrame, *column: str) -> pd.DataFrame:
    """Drop columns from a dataframe."""
    return df.drop(list(column), axis=1)


def _drop_community_level(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the community level column from the dataframe."""
    return drop_columns(df, schemas.COMMUNITY_LEVEL)


def join(
    left: pd.DataFrame, right: pd.DataFrame, key: str, strategy="left"
) -> pd.DataFrame:
    """Perform a table join."""
    return left.merge(right, on=key, how=strategy)


def _get_subcontext_df(
    level: int, report_df: pd.DataFrame, local_context_df: pd.DataFrame
) -> pd.DataFrame:
    """Get sub-community context for each community."""
    sub_report_df = _drop_community_level(_at_level(level, report_df))
    sub_context_df = _at_level(level, local_context_df)
    sub_context_df = join(sub_context_df, sub_report_df, schemas.NODE_COMMUNITY)
    sub_context_df.rename(
        columns={schemas.NODE_COMMUNITY: schemas.SUB_COMMUNITY}, inplace=True
    )
    return sub_context_df


def select(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Select columns from a dataframe."""
    return df[list(columns)]


def build_mixed_context(context: list[dict], max_tokens: int) -> str:
    """
    Build parent context by concatenating all sub-communities' contexts.

    If the context exceeds the limit, we use sub-community reports instead.
    """
    sorted_context = sorted(
        context, key=lambda x: x[schemas.CONTEXT_SIZE], reverse=True
    )

    # replace local context with sub-community reports, starting from the biggest sub-community
    substitute_reports = []
    final_local_contexts = []
    exceeded_limit = True
    context_string = ""

    for idx, sub_community_context in enumerate(sorted_context):
        if exceeded_limit:
            if sub_community_context[schemas.FULL_CONTENT]:
                substitute_reports.append(
                    {
                        schemas.COMMUNITY_ID: sub_community_context[
                            schemas.SUB_COMMUNITY
                        ],
                        schemas.FULL_CONTENT: sub_community_context[
                            schemas.FULL_CONTENT
                        ],
                    }
                )
            else:
                # this sub-community has no report, so we will use its local context
                final_local_contexts.extend(sub_community_context[schemas.ALL_CONTEXT])
                continue

            # add local context for the remaining sub-communities
            remaining_local_context = []
            for rid in range(idx + 1, len(sorted_context)):
                remaining_local_context.extend(sorted_context[rid][schemas.ALL_CONTEXT])
            new_context_string = sort_context(
                local_context=remaining_local_context + final_local_contexts,
                sub_community_reports=substitute_reports,
            )
            if num_tokens(new_context_string) <= max_tokens:
                exceeded_limit = False
                context_string = new_context_string
                break

    if exceeded_limit:
        # if all sub-community reports exceed the limit, we add reports until context is full
        substitute_reports = []
        for sub_community_context in sorted_context:
            substitute_reports.append(
                {
                    schemas.COMMUNITY_ID: sub_community_context[schemas.SUB_COMMUNITY],
                    schemas.FULL_CONTENT: sub_community_context[schemas.FULL_CONTENT],
                }
            )
            new_context_string = pd.DataFrame(substitute_reports).to_csv(
                index=False, sep=","
            )
            if num_tokens(new_context_string) > max_tokens:
                break

            context_string = new_context_string
    return context_string


def _build_mixed_context(df: pd.DataFrame, max_tokens: int) -> pd.Series:
    """Sort and trim context to fit the limit."""
    series = df[schemas.ALL_CONTEXT]
    return transform_series(
        series, lambda x: build_mixed_context(x, max_tokens=max_tokens)
    )


def _get_community_df(
    level: int,
    invalid_context_df: pd.DataFrame,
    sub_context_df: pd.DataFrame,
    community_hierarchy_df: pd.DataFrame,
    max_tokens: int,
) -> pd.DataFrame:
    """Get community context for each community."""
    # collect all sub communities' contexts for each community
    community_df = _drop_community_level(_at_level(level, community_hierarchy_df))
    invalid_community_ids = select(invalid_context_df, schemas.NODE_COMMUNITY)
    subcontext_selection = select(
        sub_context_df,
        schemas.SUB_COMMUNITY,
        schemas.FULL_CONTENT,
        schemas.ALL_CONTEXT,
        schemas.CONTEXT_SIZE,
    )

    invalid_communities = join(
        community_df, invalid_community_ids, schemas.NODE_COMMUNITY, "inner"
    )
    community_df = join(
        invalid_communities, subcontext_selection, schemas.SUB_COMMUNITY
    )
    community_df[schemas.ALL_CONTEXT] = community_df.apply(
        lambda x: {
            schemas.SUB_COMMUNITY: x[schemas.SUB_COMMUNITY],
            schemas.ALL_CONTEXT: x[schemas.ALL_CONTEXT],
            schemas.FULL_CONTENT: x[schemas.FULL_CONTENT],
            schemas.CONTEXT_SIZE: x[schemas.CONTEXT_SIZE],
        },
        axis=1,
    )
    community_df = (
        community_df.groupby(schemas.NODE_COMMUNITY)
        .agg({schemas.ALL_CONTEXT: list})
        .reset_index()
    )
    community_df[schemas.CONTEXT_STRING] = _build_mixed_context(
        community_df, max_tokens
    )
    community_df[schemas.COMMUNITY_LEVEL] = level
    return community_df


def prep_community_report_context(
    report_df: pd.DataFrame | None,
    community_hierarchy_df: pd.DataFrame,
    local_context_df: pd.DataFrame,
    level: int | str,
    max_tokens: int,
) -> pd.DataFrame:
    """
    Prep context for each community in a given level.

    For each community:
    - Check if local context fits within the limit, if yes use local context
    - If local context exceeds the limit, iteratively replace local context with sub-community reports, starting from the biggest sub-community
    """
    if report_df is None:
        report_df = pd.DataFrame()

    level = int(level)
    level_context_df = _at_level(level, local_context_df)
    valid_context_df = _within_context(level_context_df)
    invalid_context_df = _exceeding_context(level_context_df)

    # there is no report to substitute with, so we just trim the local context of the invalid context records
    # this case should only happen at the bottom level of the community hierarchy where there are no sub-communities
    if invalid_context_df.empty:
        return valid_context_df

    if report_df.empty:
        invalid_context_df[schemas.CONTEXT_STRING] = _sort_and_trim_context(
            invalid_context_df, max_tokens
        )
        set_context_size(invalid_context_df)
        invalid_context_df[schemas.CONTEXT_EXCEED_FLAG] = 0
        return union(valid_context_df, invalid_context_df)

    level_context_df = _antijoin_reports(level_context_df, report_df)

    # for each invalid context, we will try to substitute with sub-community reports
    # first get local context and report (if available) for each sub-community
    sub_context_df = _get_subcontext_df(level + 1, report_df, local_context_df)
    community_df = _get_community_df(
        level, invalid_context_df, sub_context_df, community_hierarchy_df, max_tokens
    )

    # handle any remaining invalid records that can't be subsituted with sub-community reports
    # this should be rare, but if it happens, we will just trim the local context to fit the limit
    remaining_df = _antijoin_reports(invalid_context_df, community_df)
    remaining_df[schemas.CONTEXT_STRING] = _sort_and_trim_context(
        remaining_df, max_tokens
    )

    result = union(valid_context_df, community_df, remaining_df)
    set_context_size(result)
    result[schemas.CONTEXT_EXCEED_FLAG] = 0
    return result


COMMUNITY_REPORT_MAX_INPUT_LENGTH = 8000

def _generate_report(
    send_to: Callable,
    runner: Callable,
    # cache: PipelineCache,
    # callbacks: VerbCallbacks,
    strategy: dict,
    community_id: int | str,
    community_level: int,
    community_context: str,
) -> CommunityReport | None:
    """Generate a report for a single community."""



    return runner(send_to, community_id, community_context, community_level, strategy)


# def load_strategy(
#     strategy: CreateCommunityReportsStrategyType,
# ) -> CommunityReportsStrategy:
#     """Load strategy method definition."""
#     match strategy:
#         case CreateCommunityReportsStrategyType.graph_intelligence:
#             from .strategies.graph_intelligence import run

#             return run
#         case _:
#             msg = f"Unknown strategy: {strategy}"
#             raise ValueError(msg)


def create_community_reports(
    send_to: Callable,
    local_contexts: pd.DataFrame,
    nodes: pd.DataFrame,
    community_hierarchy: pd.DataFrame,
    # callbacks: VerbCallbacks,
    # cache: PipelineCache,
    strategy: dict,
    # async_mode: AsyncType = AsyncType.AsyncIO,
    # num_threads: int = 4,
    **_kwargs,
) -> pd.DataFrame:
    """Generate entities for each row, and optionally a graph of those entities."""
    # log.debug("create_community_reports strategy=%s", strategy)

    levels = get_levels(nodes)
    reports: list[CommunityReport | None] = []
    runner = _run_extractor

    for level in levels:
        level_contexts = prep_community_report_context(
            pd.DataFrame(reports),
            local_context_df=local_contexts,
            community_hierarchy_df=community_hierarchy,
            level=level,
            max_tokens=strategy.get(
                "max_input_tokens", COMMUNITY_REPORT_MAX_INPUT_LENGTH
            ),
        )

        def run_generate(record):
            result = _generate_report(
                send_to,
                runner,
                community_id=record[schemas.NODE_COMMUNITY],
                community_level=record[schemas.COMMUNITY_LEVEL],
                community_context=record[schemas.CONTEXT_STRING],
                # cache=cache,
                # callbacks=callbacks,
                strategy=strategy,
            )
            # tick()
            return result

        local_reports = []
        for _, record in level_contexts.iterrows():
            report = run_generate(record)
            local_reports.append(report)

        # local_reports = derive_from_rows(
        #     level_contexts,
        #     run_generate,
        #     callbacks=NoopVerbCallbacks(),
        #     num_threads=num_threads,
        #     scheduling_type=async_mode,
        # )
        reports.extend([lr for lr in local_reports if lr is not None])

    return pd.DataFrame(reports)
