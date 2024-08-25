import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict
import networkx as nx

from ..llm.send import ChatLLM
from ..template.graph_extract import GRAPH_EXTRACTION_PROMPT
from ..text.text_splitter import create_text_splitter
from ..generators.graph_extactor import GraphExtractor
from . import defaults as defs


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
    send_to: ChatLLM,
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
    llm_args: Dict | None = {},
) -> EntityExtractionResult:
    """Run the entity extraction chain."""

    text_splitter = create_text_splitter(
        prechunked, chunk_size, chunk_overlap, encoding_name
    )

    extractor = GraphExtractor(
        send_to=send_to,
        prompt=extraction_prompt,
        encoding_model=encoding_model,
        max_gleanings=max_gleanings,
        llm_args=llm_args,
    )

    text_list = [doc.text.strip() for doc in docs]

    # If it's not pre-chunked, then re-chunk the input
    if not prechunked:
        text_list = text_splitter.split_text("\n".join(text_list))

    results = extractor(
        texts=list(text_list),
        # send_to=send_to,
        prompt_variables={
            "entity_types": entity_types,
            "tuple_delimiter": tuple_delimiter,
            "record_delimiter": record_delimiter,
            "completion_delimiter": completion_delimiter,
        },
        # extraction_prompt=extraction_prompt,
        # max_gleaning=max_gleanings,
    )

    graph = results.output

    if len(graph.nodes) == 0:
        raise ValueError("Graph has no nodes")
    
    if len(graph.edges) == 0:
        raise ValueError("Graph has no edges")

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


def entity_extract(
    input: pd.DataFrame,
    send_to: ChatLLM,
    column: str,
    id_column: str,
    to: str,
    graph_to: str | None = None,
    entity_types=defs.DEFAULT_ENTITY_TYPES,
    extraction_prompt=GRAPH_EXTRACTION_PROMPT,
    llm_args: Dict | None = {},
) -> pd.DataFrame:

    if entity_types is None:
        entity_types = defs.DEFAULT_ENTITY_TYPES
    output = input
    num_started = 0

    def run_strategy(row):  # -> how the text is extracted?
        nonlocal num_started
        text = row[column]
        id = row[id_column]
        result = run_extract_entities(  # calling run_gi in graph intelligence.
            docs=[Document(text=text, id=id)],
            entity_types=entity_types,
            # callbacks,
            # cache,
            send_to=send_to,
            extraction_prompt=extraction_prompt,
            llm_args=llm_args,
        )
        num_started += 1
        return [result.entities, result.graphml_graph]

    results = [run_strategy(row) for _, row in output.iterrows()]

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
