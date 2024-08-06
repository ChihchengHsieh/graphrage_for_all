import pandas as pd
from typing import Any, Callable, List, Dict, NamedTuple
from dataclasses import dataclass
import networkx as nx

{
    "stagger": 0.3,
    "num_threads": 50,
    "async_mode": "<AsyncType.Threaded: 'threaded'>",
    "strategy": {
        "type": "graph_intelligence",
        "llm": {
            "type": "openai_chat",
            "model": "gpt-4-turbo-preview",
            "max_tokens": 4000,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "request_timeout": 180.0,
            "api_base": None,
            "api_version": None,
            "organization": None,
            "proxy": None,
            "cognitive_services_endpoint": None,
            "deployment_name": None,
            "model_supports_json": True,
            "tokens_per_minute": 0,
            "requests_per_minute": 0,
            "max_retries": 10,
            "max_retry_wait": 10.0,
            "sleep_on_rate_limit_recommendation": True,
            "concurrent_requests": 25,
        },
        "stagger": 0.3,
        "num_threads": 50,
        "summarize_prompt": "\nYou are a helpful assistant responsible for generating a comprehensive summary of the data provided below.\nGiven one or two entities, and a list of descriptions, all related to the same entity or group of entities.\nPlease concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.\nIf the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.\nMake sure it is written in third person, and include the entity names so we the have full context.\n\n#######\n-Data-\nEntities: {entity_name}\nDescription List: {description_list}\n#######\nOutput:\n",
        "max_summary_length": 500,
    },
}


class DescriptionSummarizeRow(NamedTuple):
    """DescriptionSummarizeRow class definition."""

    graph: Any


def load_graph(graphml: str | nx.Graph) -> nx.Graph:
    """Load a graph from a graphml file or a networkx graph."""
    return nx.parse_graphml(graphml) if isinstance(graphml, str) else graphml


def summarize_descriptions(
    input: pd.DataFrame,
    # cache: PipelineCache,
    # callbacks: VerbCallbacks,
    column: str,
    to: str,
    send_to: Callable[[List[Dict[str, str]]], str],
    strategy: dict[str, Any] | None = None,
    **kwargs,
) -> pd.DataFrame:
    output = input
    strategy = strategy or {}
    strategy_config = {**strategy}

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
        results = run_gi(
            graph_item,
            descriptions,
            # callbacks,
            # cache,
            send_to=send_to,
            args=strategy_config,
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


@dataclass
class SummarizedDescriptionResult:
    """Entity summarization result class definition."""

    items: str | tuple[str, str]
    description: str


def run_gi(
    described_items: str | tuple[str, str],
    descriptions: list[str],
    # reporter: VerbCallbacks,
    # pipeline_cache: PipelineCache,
    send_to: Callable[[List[Dict[str, str]]], str],
    args: StrategyConfig,
) -> SummarizedDescriptionResult:
    """Run the graph intelligence entity extraction strategy."""
    return run_summarize_descriptions(send_to, described_items, descriptions, args)


from .summarize_extractor import SummarizeExtractor


def run_summarize_descriptions(
    send_to: Callable[[List[Dict[str, str]]], str],
    items: str | tuple[str, str],
    descriptions: list[str],
    # reporter: VerbCallbacks,
    args: StrategyConfig,
    entity_name_key: str = "entity_name",
    input_descriptions_key: str = "description_list",
    summarize_prompt: str | None = None,
    max_tokens=4000,
    max_summary_length=500,
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
        llm_args=args.get("llm", None),
    )

    result = extractor(items=items, descriptions=descriptions)
    return SummarizedDescriptionResult(
        items=result.items, description=result.description
    )
