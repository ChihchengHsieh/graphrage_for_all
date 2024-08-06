from dataclasses import dataclass
from typing import Any, Callable, List, Dict
import pandas as pd
import networkx as nx
from .template import GRAPH_EXTRACTION_PROMPT
from .graph_extractor import graph_extractor_forward
from .text_splitter import create_text_splitter


@dataclass
class Document:
    """Document class definition."""

    text: str
    id: str


DEFAULT_ENTITY_TYPES = ["disease", "symptom"]
EntityTypes = list[str]
ExtractedEntity = dict[str, Any]


@dataclass
class EntityExtractionResult:
    """Entity extraction result class definition."""

    entities: list[ExtractedEntity]
    graphml_graph: str | None


def run_extract_entities(
    send_to: Callable[[List[Dict[str, str]]], str],
    docs: list[Document],
    entity_types: EntityTypes,
    # default params
    prechunked=True,
    chunk_size=1200,
    chunk_overlap=100,
    extraction_prompt=GRAPH_EXTRACTION_PROMPT,
    encoding_model="cl100k_base",
    encoding_name="cl100k_base",
    max_gleanings=1,
    tuple_delimiter=None,
    record_delimiter=None,
    completion_delimiter=None,
) -> EntityExtractionResult:
    """Run the entity extraction chain."""
    # encoding_name = args.get("encoding_name", "cl100k_base")
    # # Chunking Arguments
    # prechunked = args.get("prechunked", False)
    # chunk_size = args.get("chunk_size", defs.CHUNK_SIZE)
    # chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)

    # # Extraction Arguments
    # tuple_delimiter = args.get("tuple_delimiter", None)
    # record_delimiter = args.get("record_delimiter", None)
    # completion_delimiter = args.get("completion_delimiter", None)
    # encoding_model = args.get("encoding_name", None)
    # max_gleanings = args.get("max_gleanings", defs.ENTITY_EXTRACTION_MAX_GLEANINGS)

    # note: We're not using UnipartiteGraphChain.from_params
    # because we want to pass "timeout" to the llm_kwargs

    text_splitter = create_text_splitter(
        prechunked, chunk_size, chunk_overlap, encoding_name
    )

    # extractor = GraphExtractor(
    #     llm_invoker=llm,
    #     prompt=extraction_prompt,
    #     encoding_model=encoding_model,
    #     max_gleanings=max_gleanings,
    #     on_error=lambda e, s, d: (
    #         reporter.error("Entity Extraction Error", e, s, d) if reporter else None
    #     ),
    # )
    text_list = [doc.text.strip() for doc in docs]

    # If it's not pre-chunked, then re-chunk the input
    if not prechunked:
        text_list = text_splitter.split_text("\n".join(text_list))

    results = graph_extractor_forward(
        texts=list(text_list),
        send_to=send_to,
        prompt_variables={
            "entity_types": entity_types,
            "tuple_delimiter": tuple_delimiter,
            "record_delimiter": record_delimiter,
            "completion_delimiter": completion_delimiter,
        },
        extraction_prompt=extraction_prompt,
        max_gleaning=max_gleanings,
    )

    graph = results.output
    # Map the "source_id" back to the "id" field
    for _, node in graph.nodes(data=True):  # type: ignore
        if node is not None:
            node["source_id"] = ",".join(
                docs[int(id)].id for id in node["source_id"].split(",")
            )

    for _, _, edge in graph.edges(data=True):  # type: ignore
        if edge is not None:
            edge["source_id"] = ",".join(
                docs[int(id)].id for id in edge["source_id"].split(",")
            )

    entities = [
        ({"name": item[0], **(item[1] or {})})
        for item in graph.nodes(data=True)
        if item is not None
    ]

    graph_data = "".join(nx.generate_graphml(graph))
    return EntityExtractionResult(entities, graph_data)


def run_gi(
    docs: list[Document],
    entity_types: EntityTypes,
    # reporter: VerbCallbacks,
    # pipeline_cache: PipelineCache,
    # args: StrategyConfig,
    send_to: Callable[[List[Dict[str, str]]], str],
    extraction_prompt: str,
) -> EntityExtractionResult:
    """Run the graph intelligence entity extraction strategy."""
    return run_extract_entities(
        send_to,
        docs,
        entity_types,
        extraction_prompt,
    )


def entity_extract(
    input: pd.DataFrame,
    # cache: PipelineCache,
    # callbacks: VerbCallbacks,
    send_to: Callable[[List[Dict[str, str]]], str],
    column: str,
    id_column: str,
    to: str,
    graph_to: str | None = None,
    # async_mode: AsyncType = AsyncType.AsyncIO,
    entity_types=DEFAULT_ENTITY_TYPES,
    extraction_prompt=GRAPH_EXTRACTION_PROMPT,
) -> pd.DataFrame:

    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    output = input
    num_started = 0

    def run_strategy(row):  # -> how the text is extracted?
        nonlocal num_started
        text = row[column]
        id = row[id_column]
        result = run_gi(  # calling run_gi in graph intelligence.
            [Document(text=text, id=id)],
            entity_types,
            # callbacks,
            # cache,
            send_to=send_to,
            extraction_prompt=extraction_prompt,
        )
        num_started += 1
        return [result.entities, result.graphml_graph]

    results = []
    for _, row in output.iterrows():
        result = run_strategy(row)
        results.append(result)

    to_result = []
    graph_to_result = []
    for result in results:
        if result:
            to_result.append(result[0])
            graph_to_result.append(result[1])
        else:
            to_result.append(None)
            graph_to_result.append(None)

    output[to] = to_result
    if graph_to is not None:
        output[graph_to] = graph_to_result

    return output.reset_index(drop=True)
