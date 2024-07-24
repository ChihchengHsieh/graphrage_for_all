from secret import OPENAI_API_KEY
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Callable, Collection, Iterable, Literal, cast
import pandas as pd
import tiktoken
import argparse

import numbers
import html
import re
import networkx as nx
from typing import Dict, List
import openai

openai.api_key = OPENAI_API_KEY


def get_args_parser():
    parser = argparse.ArgumentParser("Indexing script", add_help=False)
    return parser


LengthFn = Callable[[str], int]
EncodedText = list[int]
DecodeFn = Callable[[EncodedText], str]
EncodeFn = Callable[[str], EncodedText]

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100


max_gleanings = 1
join_descriptions = True
input_text_key = "input_text"
tuple_delimiter_key = "tuple_delimiter"
record_delimiter_key = "record_delimiter"
completion_delimiter_key = "completion_delimiter"
entity_types_key = "entity_types"

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities were missed in the last extraction.  Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.\n"
extraction_prompt = GRAPH_EXTRACTION_PROMPT


class TextSplitter(ABC):
    """Text splitter class definition."""

    _chunk_size: int
    _chunk_overlap: int
    _length_function: LengthFn
    _keep_separator: bool
    _add_start_index: bool
    _strip_whitespace: bool

    def __init__(
        self,
        # based on text-ada-002-embedding max input buffer length
        # https://platform.openai.com/docs/guides/embeddings/second-generation-models
        chunk_size: int = 8191,
        chunk_overlap: int = 100,
        length_function: LengthFn = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ):
        """Init method definition."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split text method definition."""


@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: DecodeFn
    """ Function to decode a list of token ids to a string"""
    encode: EncodeFn
    """ Function to encode a string to a list of token ids"""


class TokenTextSplitter(TextSplitter):
    """Token text splitter class definition."""

    _allowed_special: Literal["all"] | set[str]
    _disallowed_special: Literal["all"] | Collection[str]

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: str | None = None,
        allowed_special: Literal["all"] | set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ):
        """Init method definition."""
        super().__init__(**kwargs)
        if model_name is not None:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                enc = tiktoken.get_encoding(encoding_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special or set()
        self._disallowed_special = disallowed_special

    def encode(self, text: str) -> list[int]:
        """Encode the given text into an int-vector."""
        return self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.encode(text))

    def split_text(self, text: str | list[str]) -> list[str]:
        """Split text method."""
        if cast(bool, pd.isna(text)) or text == "":
            return []
        if isinstance(text, list):
            text = " ".join(text)
        if not isinstance(text, str):
            msg = f"Attempting to split a non-string value, actual is {type(text)}"
            raise TypeError(msg)

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=lambda text: self.encode(text),
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)


def split_text_on_tokens(text: str, tokenizer: Tokenizer) -> list[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: list[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits


def create_text_splitter(
    chunk_size: int, chunk_overlap: int, encoding_name: str
) -> TextSplitter:
    """Create a text splitter for the extraction chain.

    Args:
        - prechunked - Whether the text is already chunked
        - chunk_size - The size of each chunk
        - chunk_overlap - The overlap between chunks
        - encoding_name - The name of the encoding to use
    Returns:
        - output - A text splitter
    """

    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)  # for exponential backoff


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10),
)
def chat_completion_with_backoff(**kwargs):
    return openai.chat.completions.create(**kwargs)

    # prompt_variables = {
    #     **prompt_variables,
    #     self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
    #     or DEFAULT_TUPLE_DELIMITER,
    #     self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
    #     or DEFAULT_RECORD_DELIMITER,
    #     self._completion_delimiter_key: prompt_variables.get(
    #         self._completion_delimiter_key
    #     )
    #     or DEFAULT_COMPLETION_DELIMITER,
    #     self._entity_types_key: ",".join(
    #         prompt_variables[self._entity_types_key] or DEFAULT_ENTITY_TYPES
    #     ),
    # }


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

    # prompt_variables = {
    #     **prompt_variables,
    #     self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
    #     or DEFAULT_TUPLE_DELIMITER,
    #     self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
    #     or DEFAULT_RECORD_DELIMITER,
    #     self._completion_delimiter_key: prompt_variables.get(
    #         self._completion_delimiter_key
    #     )
    #     or DEFAULT_COMPLETION_DELIMITER,
    #     self._entity_types_key: ",".join(
    #         prompt_variables[self._entity_types_key] or DEFAULT_ENTITY_TYPES
    #     ),
    # }


def send_to_open_ai(messages):
    response = chat_completion_with_backoff(
        **{
            "model": "gpt-3.5-turbo",
            "messages": messages,
        }
    )

    output = response.choices[0].message.content
    return output


def execute_llm(
    extraction_prompt,
    send_to: Callable[[List[Dict[str, str]]], str],
    variables: Dict | None = None,
    history: List | None = None,
):
    input = perform_variable_replacements(extraction_prompt, history, variables)
    messages = []
    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": input,
        }
    )

    response = send_to(messages) # modify this for other llms.

    output = response.choices[0].message.content

    history = [*history, {"role": "system", "content": output}]

    return output, history


def process_document(
    extraction_prompt: str, text: str, prompt_variables: dict[str, str]
) -> str:

    #  execute_llm(prompt, variables, history: List | None):

    results, history = execute_llm(
        extraction_prompt=extraction_prompt,
        variables={
            **prompt_variables,
            input_text_key: text,
        },
        history=[],
    )

    results = results or ""
    # Repeat to ensure we maximize entity count

    for i in range(max_gleanings):

        glean_response, history = execute_llm(
            CONTINUE_PROMPT,
            # name=f"extract-continuation-{i}",
            history=history or [],
        )

        results += glean_response or ""

        # if this is the final glean, don't bother updating the continuation flag
        if i >= max_gleanings - 1:
            break

        continuation = execute_llm(
            LOOP_PROMPT,
            # name=f"extract-loopcheck-{i}",
            history=history or [],
        )

        if continuation.output != "YES":
            break

    return results


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


DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    output: nx.Graph
    source_docs: dict[Any, Any]


def graph_extractor_forward(
    texts: list[str], prompt_variables: dict[str, Any] | None = None
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
            GRAPH_EXTRACTION_PROMPT,
            text,
            prompt_variables,
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


from retreivers.radiowiki import RadioWikiRetriever

from dataclasses import dataclass


@dataclass
class Document:
    """Document class definition."""

    text: str
    title: str
    id: str


from langchain_core.documents.base import Document as LC_doc
from typing import List


def lc_to_graphrag_doc(lc_docs: List[LC_doc]):
    return [
        Document(title=d.metadata["title"], text=d.page_content, id=i)
        for i, d in enumerate(lc_docs)
    ]


def lc_doc_to_df(lc_docs: List[LC_doc]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": str(i),
                "text": d.page_content,
                "title": d.metadata["title"],
            }
            for i, d in enumerate(lc_docs)
        ]
    )


DEFAULT_ENTITY_TYPES = [
    "disease",
    "symptom",
]
entity_types = DEFAULT_ENTITY_TYPES


# input is created using:
@dataclass
class EntityExtractionResult:
    """Entity extraction result class definition."""

    entities: list[dict[str, Any]]
    graphml_graph: str | None


def run_extract_entities(
    row_docs: List[Document],
    entity_types: List[str],
):
    text_splitter = create_text_splitter(CHUNK_SIZE, CHUNK_OVERLAP, "cl100k_base")
    text_list = [doc.text.strip() for doc in row_docs]
    text_list = text_splitter.split_text("\n".join(text_list))
    results = graph_extractor_forward(
        list(text_list),
        {
            "entity_types": entity_types,
        },
    )

    # raise StopIteration()

    graph = results.output
    # Map the "source_id" back to the "id" field
    for _, node in graph.nodes(data=True):  # type: ignore
        if node is not None:
            node["source_id"] = ",".join(
                row_docs[int(id)].id for id in node["source_id"].split(",")
            )
    # their documents has ids
    for _, _, edge in graph.edges(data=True):  # type: ignore
        if edge is not None:
            edge["source_id"] = ",".join(
                row_docs[int(id)].id for id in edge["source_id"].split(",")
            )

    entities = [
        ({"name": item[0], **(item[1] or {})})
        for item in graph.nodes(data=True)
        if item is not None
    ]

    graph_data = "".join(nx.generate_graphml(graph))
    return EntityExtractionResult(entities, graph_data)


def run_strategy(row):
    text = row["text"]
    id = row["id"]
    title = row["title"]
    result = run_extract_entities(
        [Document(text=text, id=id, title=title)],
        entity_types,
    )
    return [result.entities, result.graphml_graph]


to = "to_col"
graph_to = "graph_to_col"


def workflow_run(docs_df):
    to_result = []
    graph_to_result = []
    output = docs_df  # I guess?
    results = []

    for idx, row in docs_df.iterrows():
        result = run_strategy(row)
        results.append(result)

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

    return output


from pathlib import Path


def join_path(file_path: str, file_name: str) -> Path:
    """Join a path and a file. Independent of the OS."""
    return Path(file_path) / Path(file_name).parent / Path(file_name).name


def file_storage_set(
    file_path: str,
    value: Any,
) -> None:
    """Set method definition."""
    is_bytes = isinstance(value, bytes)
    write_type = "wb" if is_bytes else "w"
    encoding = None if is_bytes else "utf-8"
    with open(file_path, cast(Any, write_type), encoding=encoding) as f:
        f.write(value)


def parquet_table_emitter_emit(name: str, data: pd.DataFrame) -> None:
    """Emit a dataframe to storage."""
    filename = f"{name}.parquet"
    file_storage_set(filename, data.to_parquet())


def main(args):

    retriever = RadioWikiRetriever()
    docs = retriever.request("atelectasis")
    docs_df = lc_doc_to_df(docs)
    output = workflow_run(docs_df)
    parquet_table_emitter_emit("atelectasis_idx", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Detection Training Script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
