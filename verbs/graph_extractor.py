import networkx as nx
from dataclasses import dataclass
from typing import Any, Callable, List, Dict
from .template import GRAPH_EXTRACTION_PROMPT, CONTINUE_PROMPT, LOOP_PROMPT
import html
import numbers
import re
import tiktoken
from .llm import execute_llm


encoding = tiktoken.get_encoding("cl100k_base")
yes = encoding.encode("YES")
no = encoding.encode("NO")
loop_args = {"logit_bias": {yes[0]: 100, no[0]: 100}, "max_tokens": 1}

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    output: nx.Graph
    source_docs: dict[Any, Any]


join_descriptions = True
input_text_key = "input_text"
tuple_delimiter_key = "tuple_delimiter"
record_delimiter_key = "record_delimiter"
completion_delimiter_key = "completion_delimiter"
entity_types_key = "entity_types"
max_gleaning = 1


def perform_variable_replacements(
    input: str, history: list[dict], variables: dict | None
) -> str:
    """Perform variable replacements on the input string and in a chat log."""
    result = input

    def replace_all(input: str) -> str:
        result = input
        if variables:
            for entry in variables:
                result = result.replace(f"{{{entry}}}", variables[entry])
        return result

    result = replace_all(result)
    for i in range(len(history)):
        entry = history[i]
        if entry.get("role") == "system":
            history[i]["content"] = replace_all(entry.get("content") or "")

    return result


def graph_extractor_forward(
    texts: list[str],
    send_to: Callable[[List[Dict[str, str]]], str],
    prompt_variables: dict[str, Any] | None = None,
    max_gleaning: int = 1,
    extraction_prompt: str = GRAPH_EXTRACTION_PROMPT,
) -> GraphExtractionResult:
    """Call method definition."""

    if prompt_variables is None:

        prompt_variables = {}

    all_records: dict[int, str] = {}

    source_doc_map: dict[int, str] = {}

    # Wire defaults into the prompt variables

    prompt_variables = {
        **prompt_variables,
        tuple_delimiter_key: prompt_variables.get(tuple_delimiter_key)
        or DEFAULT_TUPLE_DELIMITER,
        record_delimiter_key: prompt_variables.get(record_delimiter_key)
        or DEFAULT_RECORD_DELIMITER,
        completion_delimiter_key: prompt_variables.get(completion_delimiter_key)
        or DEFAULT_COMPLETION_DELIMITER,
        entity_types_key: ",".join(
            prompt_variables[entity_types_key] or DEFAULT_ENTITY_TYPES
        ),
    }

    for doc_index, text in enumerate(texts):
        # Invoke the entity extraction
        result = process_document(
            extraction_prompt,
            text,
            prompt_variables,
            max_gleanings=max_gleaning,
            send_to=send_to,
        )
        source_doc_map[doc_index] = text
        all_records[doc_index] = result

    output = process_results(
        all_records,
        prompt_variables.get(tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER),
        prompt_variables.get(record_delimiter_key, DEFAULT_RECORD_DELIMITER),
    )

    return GraphExtractionResult(
        output=output,
        source_docs=source_doc_map,
    )


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def unpack_descriptions(data) -> list[str]:
    value = data.get("description", None)
    return [] if value is None else value.split("\n")


def _unpack_source_ids(data) -> list[str]:
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")


def process_document(
    extraction_prompt: str,
    text: str,
    prompt_variables: dict[str, str],
    max_gleanings: int,
    send_to: Callable[[List[Dict[str, str]]], str],
) -> str:

    #  execute_llm(prompt, variables, history: List | None):

    results, history = execute_llm(
        input=perform_variable_replacements(
            extraction_prompt,
            [],
            variables={
                **prompt_variables,
                input_text_key: text,
            },
        ),
        send_to=send_to,
        # extraction_prompt=extraction_prompt,
        # variables={
        #     **prompt_variables,
        #     input_text_key: text,
        # },
        history=[],
    )

    results = results or ""
    # Repeat to ensure we maximize entity count

    for i in range(max_gleanings):

        glean_response, history = execute_llm(
            input=perform_variable_replacements(
                CONTINUE_PROMPT,
                history,
                variables={
                    **prompt_variables,
                    input_text_key: text,
                },
            ),
            send_to=send_to,
            # extraction_prompt=CONTINUE_PROMPT,
            # name=f"extract-continuation-{i}",
            history=history or [],
        )

        results += glean_response or ""

        # if this is the final glean, don't bother updating the continuation flag
        if i >= max_gleanings - 1:
            break

        continuation = execute_llm(
            input=perform_variable_replacements(
                LOOP_PROMPT,
                history,
                variables={
                    **prompt_variables,
                    input_text_key: text,
                },
            ),
            send_to=send_to,
            # name=f"extract-loopcheck-{i}",
            history=history or [],
            model_args=loop_args,
        )

        if continuation != "YES":
            break

    return results


def process_results(
    results: dict[int, str],
    tuple_delimiter: str,
    record_delimiter: str,
) -> nx.Graph:
    graph = nx.Graph()
    for source_doc_id, extracted_data in results.items():
        records = [r.strip() for r in extracted_data.split(record_delimiter)]

        for record in records:
            record = re.sub(r"^\(|\)$", "", record.strip())
            record_attributes = record.split(tuple_delimiter)

            if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                # add this record as a node in the G
                entity_name = clean_str(record_attributes[1].upper())
                entity_type = clean_str(record_attributes[2].upper())
                entity_description = clean_str(record_attributes[3])

                if entity_name in graph.nodes():
                    node = graph.nodes[entity_name]
                    if join_descriptions:
                        node["description"] = "\n".join(
                            list(
                                {
                                    *unpack_descriptions(node),
                                    entity_description,
                                }
                            )
                        )
                    else:
                        if len(entity_description) > len(node["description"]):
                            node["description"] = entity_description
                    node["source_id"] = ", ".join(
                        list(
                            {
                                *_unpack_source_ids(node),
                                str(source_doc_id),
                            }
                        )
                    )
                    node["entity_type"] = (
                        entity_type if entity_type != "" else node["entity_type"]
                    )
                else:
                    graph.add_node(
                        entity_name,
                        type=entity_type,
                        description=entity_description,
                        source_id=str(source_doc_id),
                    )

            if record_attributes[0] == '"relationship"' and len(record_attributes) >= 5:
                # add this record as edge
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                edge_source_id = clean_str(str(source_doc_id))
                weight = (
                    float(record_attributes[-1])
                    if isinstance(record_attributes[-1], numbers.Number)
                    else 1.0
                )
                if source not in graph.nodes():
                    graph.add_node(
                        source,
                        type="",
                        description="",
                        source_id=edge_source_id,
                    )
                if target not in graph.nodes():
                    graph.add_node(
                        target,
                        type="",
                        description="",
                        source_id=edge_source_id,
                    )
                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target)
                    if edge_data is not None:
                        weight += edge_data["weight"]
                        if join_descriptions:
                            edge_description = "\n".join(
                                list(
                                    {
                                        *unpack_descriptions(edge_data),
                                        edge_description,
                                    }
                                )
                            )
                        edge_source_id = ", ".join(
                            list(
                                {
                                    *_unpack_source_ids(edge_data),
                                    str(source_doc_id),
                                }
                            )
                        )
                graph.add_edge(
                    source,
                    target,
                    weight=weight,
                    description=edge_description,
                    source_id=edge_source_id,
                )

    return graph
