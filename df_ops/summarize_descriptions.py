from typing import Any, Dict, NamedTuple
from llm.send import ChatLLM

import pandas as pd
import networkx as nx

from dataclasses import dataclass
from utils.graph import load_graph
from generators.summarize_extractor import SummarizeExtractor


class DescriptionSummarizeRow(NamedTuple):
    """DescriptionSummarizeRow class definition."""

    graph: Any


@dataclass
class SummarizedDescriptionResult:
    """Entity summarization result class definition."""

    items: str | tuple[str, str]
    description: str


def run_summarize_descriptions(
    send_to: ChatLLM,
    items: str | tuple[str, str],
    descriptions: list[str],
    # reporter: VerbCallbacks,
    entity_name_key: str = "entity_name",
    input_descriptions_key: str = "description_list",
    summarize_prompt: str | None = None,
    max_tokens=4000,
    max_summary_length=500,
    llm_args: Dict = {},
) -> SummarizedDescriptionResult:
    """Run the entity extraction chain."""
    # Extraction Arguments

    extractor = SummarizeExtractor(
        send_to=send_to,
        summarization_prompt=summarize_prompt,
        entity_name_key=entity_name_key,
        input_descriptions_key=input_descriptions_key,
        # on_error=lambda e, stack, details: (
        #     reporter.error("Entity Extraction Error", e, stack, details)
        #     if reporter
        #     else None
        # ),
        max_summary_length=max_summary_length,
        max_input_tokens=max_tokens,
        llm_args=llm_args,
    )

    result = extractor(items=items, descriptions=descriptions)
    return SummarizedDescriptionResult(
        items=result.items, description=result.description
    )


def summarize_descriptions(
    input: pd.DataFrame,
    # cache: PipelineCache,
    # callbacks: VerbCallbacks,
    column: str,
    to: str,
    send_to: ChatLLM,
    max_summary_length: int = 500,
    # strategy: dict[str, Any] | None = None,
    llm_args: Dict | None = None,
) -> pd.DataFrame:
    output = input
    # strategy = strategy or {}
    # strategy_config = {**strategy}

    def get_resolved_entities(
        row,
    ):
        graph: nx.Graph = load_graph(getattr(row, column))
        # ticker_length = len(graph.nodes) + len(graph.edges)
        # ticker = progress_ticker(callbacks.progress, ticker_length)

        futures = [
            do_summarize_descriptions(
                node,
                sorted(set(graph.nodes[node].get("description", "").split("\n"))),
                # ticker,
            )
            for node in graph.nodes()
        ]
        futures += [
            do_summarize_descriptions(
                edge,
                sorted(set(graph.edges[edge].get("description", "").split("\n"))),
                # ticker,
            )
            for edge in graph.edges()
        ]

        results = futures

        for result in results:
            graph_item = result.items
            if isinstance(graph_item, str) and graph_item in graph.nodes():
                graph.nodes[graph_item]["description"] = result.description
            elif isinstance(graph_item, tuple) and graph_item in graph.edges():
                graph.edges[graph_item]["description"] = result.description

        return DescriptionSummarizeRow(
            graph="\n".join(nx.generate_graphml(graph)),
        )

    def do_summarize_descriptions(
        graph_item: str | tuple[str, str],
        descriptions: list[str],
        # ticker: ProgressTicker,
        # semaphore: asyncio.Semaphore,
    ):
        results = run_summarize_descriptions(
            max_summary_length=max_summary_length,
            send_to=send_to,
            items=graph_item,
            descriptions=descriptions,
            llm_args=llm_args,
        )
        return results

    # Graph is always on row 0, so here a derive from rows does not work
    # This iteration will only happen once, but avoids hardcoding a iloc[0]
    # Since parallelization is at graph level (nodes and edges), we can't use
    # the parallelization of the derive_from_rows
    # semaphore = asyncio.Semaphore(kwargs.get("num_threads", 4))

    results = [get_resolved_entities(row) for row in output.itertuples()]

    to_result = []

    for result in results:
        if result:
            to_result.append(result.graph)
        else:
            to_result.append(None)
    output[to] = to_result
    return output


StrategyConfig = dict[str, Any]
