from typing import Any
import tiktoken
import networkx as nx
from dataclasses import dataclass
import re
import numbers
from collections.abc import Mapping

from ..llm.send import ChatLLM, ModelArgs
from ..template.graph_extract import (
    GRAPH_EXTRACTION_PROMPT,
    CONTINUE_PROMPT,
    LOOP_PROMPT,
)
from ..df_ops import defaults as defs
from ..utils.str import clean_str
from ..llm.send import replace_and_send


@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    output: nx.Graph
    source_docs: dict[Any, Any]
    all_records: list[str]


class GraphExtractor:
    """Unipartite graph extractor class definition."""

    _send_to: ChatLLM
    _join_descriptions: bool
    _tuple_delimiter_key: str
    _record_delimiter_key: str
    _entity_types_key: str
    _input_text_key: str
    _completion_delimiter_key: str
    _entity_name_key: str
    _input_descriptions_key: str
    _extraction_prompt: str
    _summarization_prompt: str
    _loop_args: dict[str, Any]
    _max_gleanings: int
    _llm_args: ModelArgs

    def __init__(
        self,
        send_to: ChatLLM,
        tuple_delimiter_key: str | None = None,
        record_delimiter_key: str | None = None,
        input_text_key: str | None = None,
        entity_types_key: str | None = None,
        completion_delimiter_key: str | None = None,
        prompt: str | None = None,
        join_descriptions=True,
        encoding_model: str | None = None,
        max_gleanings: int | None = None,
        llm_args: dict[str, Any] | None = None,
    ):
        """Init method definition."""
        # TODO: streamline construction
        self._send_to = send_to
        self._join_descriptions = join_descriptions
        self._input_text_key = input_text_key or "input_text"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        self._entity_types_key = entity_types_key or "entity_types"
        self._extraction_prompt = prompt or GRAPH_EXTRACTION_PROMPT
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else defs.ENTITY_EXTRACTION_MAX_GLEANINGS
        )

        # Construct the looping arguments
        encoding = tiktoken.get_encoding(encoding_model or "cl100k_base")
        yes = encoding.encode("YES")
        no = encoding.encode("NO")
        self._loop_args = {"logit_bias": {yes[0]: 100, no[0]: 100}, "max_tokens": 1}
        self._llm_args = llm_args or {}

    def __call__(
        self, texts: list[str], prompt_variables: dict[str, Any] | None = None
    ) -> GraphExtractionResult:
        """Call method definition."""
        if prompt_variables is None:
            prompt_variables = {}
        all_records: dict[int, str] = {}
        source_doc_map: dict[int, str] = {}

        # Wire defaults into the prompt variables
        prompt_variables = {
            **prompt_variables,
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
            or defs.DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
            or defs.DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: prompt_variables.get(
                self._completion_delimiter_key
            )
            or defs.DEFAULT_COMPLETION_DELIMITER,
            self._entity_types_key: ",".join(
                prompt_variables[self._entity_types_key] or defs.DEFAULT_ENTITY_TYPES
            ),
        }

        for doc_index, text in enumerate(texts):
            # try:
            # Invoke the entity extraction
            result = self._process_document(text, prompt_variables)
            source_doc_map[doc_index] = text
            all_records[doc_index] = result
            # except Exception as e:
            #     print("error extracting graph")

        output = self._process_results(
            all_records,
            prompt_variables.get(
                self._tuple_delimiter_key, defs.DEFAULT_TUPLE_DELIMITER
            ),
            prompt_variables.get(
                self._record_delimiter_key, defs.DEFAULT_RECORD_DELIMITER
            ),
        )

        return GraphExtractionResult(
            output=output,
            source_docs=source_doc_map,
            all_records=all_records,
        )

    def _process_document(self, text: str, prompt_variables: dict[str, str]) -> str:

        response = replace_and_send(
            send_to=self._send_to,
            template=self._extraction_prompt,
            history=[],
            replacing_variable={
                **prompt_variables,
                self._input_text_key: re.sub(r"[^a-zA-Z0-9\s]", "", text),
            },
            llm_args=self._llm_args,
        )
        # response = await self._llm(
        #     self._extraction_prompt,
        #     variables={
        #         **prompt_variables,
        #         self._input_text_key: text,
        #     },
        # )

        results = response.output or ""

        # Repeat to ensure we maximize entity count
        for i in range(self._max_gleanings):
            # glean_response = await self._send_to(
            #     CONTINUE_PROMPT,
            #     name=f"extract-continuation-{i}",
            #     history=response.history or [],
            # )

            glean_response = replace_and_send(
                send_to=self._send_to,
                template=CONTINUE_PROMPT,
                history=response.history or [],
                llm_args=self._llm_args,
            )
            results += glean_response.output or ""

            # if this is the final glean, don't bother updating the continuation flag
            if i >= self._max_gleanings - 1:
                break

            # continuation = await self._send_to(
            #     LOOP_PROMPT,
            #     name=f"extract-loopcheck-{i}",
            #     history=glean_response.history or [],
            #     model_parameters=self._loop_args,
            # )
            continuation = replace_and_send(
                send_to=self._send_to,
                template=LOOP_PROMPT,
                history=glean_response.history or [],
                llm_args=self._loop_args,
            )
            if continuation.output != "YES":
                break

        return results

    def _process_results(
        self,
        results: dict[int, str],
        tuple_delimiter: str,
        record_delimiter: str,
    ) -> nx.Graph:
        """Parse the result string to create an undirected unipartite graph.

        Args:
            - results - dict of results from the extraction chain
            - tuple_delimiter - delimiter between tuples in an output record, default is '<|>'
            - record_delimiter - delimiter between records, default is '##'
        Returns:
            - output - unipartite graph in graphML format
        """
        graph = nx.Graph()
        for source_doc_id, extracted_data in results.items():
            records = [r.strip() for r in extracted_data.split(record_delimiter)]

            for i, record in enumerate(records):
                record = re.sub(r"^\(|\)$", "", record.strip())
                record_attributes = record.split(tuple_delimiter)

                if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                    # add this record as a node in the G
                    entity_name = clean_str(record_attributes[1].upper())
                    entity_type = clean_str(record_attributes[2].upper())
                    entity_description = clean_str(record_attributes[3])

                    if entity_name in graph.nodes():
                        node = graph.nodes[entity_name]
                        if self._join_descriptions:
                            node["description"] = "\n".join(
                                list(
                                    {
                                        *_unpack_descriptions(node),
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

                if record_attributes[0] == '"relationship"':
                    if len(record_attributes) >= 4:
                        if len(record_attributes) == 4:
                            record_attributes.append(1.0)
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
                                if self._join_descriptions:
                                    edge_description = "\n".join(
                                        list(
                                            {
                                                *_unpack_descriptions(edge_data),
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


def _unpack_descriptions(data: Mapping) -> list[str]:
    value = data.get("description", None)
    return [] if value is None else value.split("\n")


def _unpack_source_ids(data: Mapping) -> list[str]:
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")
