# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Indexing-Engine to Query Read Adapters.

The parts of these functions that do type adaptation, renaming, collating, etc. should eventually go away.
Ideally this is just a straight read-thorugh into the object model.
"""

from typing import Callable, cast
import pandas as pd
from typing import Any
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def to_str(data: pd.Series, column_name: str | None) -> str:
    """Convert and validate a value to a string."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    if column_name in data:
        return str(data[column_name])
    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)


def to_optional_str(data: pd.Series, column_name: str | None) -> str | None:
    """Convert and validate a value to an optional string."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    if column_name in data:
        value = data[column_name]
        if value is None:
            return None
        return str(data[column_name])
    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)


def to_list(
    data: pd.Series, column_name: str | None, item_type: type | None = None
) -> list:
    """Convert and validate a value to a list."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    if column_name in data:
        value = data[column_name]
        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not isinstance(value, list):
            msg = f"value is not a list: {value} ({type(value)})"
            raise ValueError(msg)

        if item_type is not None:
            for v in value:
                if not isinstance(v, item_type):
                    msg = f"list item has item that is not {item_type}: {v} ({type(v)})"
                    raise TypeError(msg)
        return value

    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)


def to_optional_list(
    data: pd.Series, column_name: str | None, item_type: type | None = None
) -> list | None:
    """Convert and validate a value to an optional list."""
    if column_name is None:
        return None

    if column_name in data:
        value = data[column_name]  # type: ignore
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not isinstance(value, list):
            msg = f"value is not a list: {value} ({type(value)})"
            raise ValueError(msg)

        if item_type is not None:
            for v in value:
                if not isinstance(v, item_type):
                    msg = f"list item has item that is not {item_type}: {v} ({type(v)})"
                    raise TypeError(msg)
        return value

    return None


def to_int(data: pd.Series, column_name: str | None) -> int:
    """Convert and validate a value to an int."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    if column_name in data:
        value = data[column_name]
        if isinstance(value, float):
            value = int(value)
        if not isinstance(value, int):
            msg = f"value is not an int: {value} ({type(value)})"
            raise ValueError(msg)
    else:
        msg = f"Column {column_name} not found in data"
        raise ValueError(msg)

    return int(value)


def to_optional_int(data: pd.Series, column_name: str | None) -> int | None:
    """Convert and validate a value to an optional int."""
    if column_name is None:
        return None

    if column_name in data:
        value = data[column_name]

        if value is None:
            return None

        if isinstance(value, float):
            value = int(value)
        if not isinstance(value, int):
            msg = f"value is not an int: {value} ({type(value)})"
            raise ValueError(msg)
    else:
        msg = f"Column {column_name} not found in data"
        raise ValueError(msg)

    return int(value)


def to_float(data: pd.Series, column_name: str | None) -> float:
    """Convert and validate a value to a float."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    if column_name in data:
        value = data[column_name]
        if not isinstance(value, float):
            msg = f"value is not a float: {value} ({type(value)})"
            raise ValueError(msg)
    else:
        msg = f"Column {column_name} not found in data"
        raise ValueError(msg)

    return float(value)


def to_optional_float(data: pd.Series, column_name: str | None) -> float | None:
    """Convert and validate a value to an optional float."""
    if column_name is None:
        return None

    if column_name in data:
        value = data[column_name]
        if value is None:
            return None
        if not isinstance(value, float):
            msg = f"value is not a float: {value} ({type(value)})"
            raise ValueError(msg)
    else:
        msg = f"Column {column_name} not found in data"
        raise ValueError(msg)

    return float(value)


def to_dict(
    data: pd.Series,
    column_name: str | None,
    key_type: type | None = None,
    value_type: type | None = None,
) -> dict:
    """Convert and validate a value to a dict."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    if column_name in data:
        value = data[column_name]
        if not isinstance(value, dict):
            msg = f"value is not a dict: {value} ({type(value)})"
            raise ValueError(msg)

        if key_type is not None:
            for v in value:
                if not isinstance(v, key_type):
                    msg = f"dict key has item that is not {key_type}: {v} ({type(v)})"
                    raise TypeError(msg)

        if value_type is not None:
            for v in value.values():
                if not isinstance(v, value_type):
                    msg = (
                        f"dict value has item that is not {value_type}: {v} ({type(v)})"
                    )
                    raise TypeError(msg)
        return value

    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)


def to_optional_dict(
    data: pd.Series,
    column_name: str | None,
    key_type: type | None = None,
    value_type: type | None = None,
) -> dict | None:
    """Convert and validate a value to an optional dict."""
    if column_name is None:
        return None

    if column_name in data:
        value = data[column_name]
        if value is None:
            return None
        if not isinstance(value, dict):
            msg = f"value is not a dict: {value} ({type(value)})"
            raise TypeError(msg)

        if key_type is not None:
            for v in value:
                if not isinstance(v, key_type):
                    msg = f"dict key has item that is not {key_type}: {v} ({type(v)})"
                    raise TypeError(msg)

        if value_type is not None:
            for v in value.values():
                if not isinstance(v, value_type):
                    msg = (
                        f"dict value has item that is not {value_type}: {v} ({type(v)})"
                    )
                    raise TypeError(msg)

        return value

    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)


@dataclass
class Identified:
    """A protocol for an item with an ID."""

    id: str
    """The ID of the item."""

    short_id: str | None
    """Human readable ID used to refer to this community in prompts or texts displayed to users, such as in a report text (optional)."""


@dataclass
class Named(Identified):
    """A protocol for an item with a name/title."""

    title: str
    """The name/title of the item."""


@dataclass
class TextUnit(Identified):
    """A protocol for a TextUnit item in a Document database."""

    text: str
    """The text of the unit."""

    text_embedding: list[float] | None = None
    """The text embedding for the text unit (optional)."""

    entity_ids: list[str] | None = None
    """List of entity IDs related to the text unit (optional)."""

    relationship_ids: list[str] | None = None
    """List of relationship IDs related to the text unit (optional)."""

    covariate_ids: dict[str, list[str]] | None = None
    "Dictionary of different types of covariates related to the text unit (optional)."

    n_tokens: int | None = None
    """The number of tokens in the text (optional)."""

    document_ids: list[str] | None = None
    """List of document IDs in which the text unit appears (optional)."""

    attributes: dict[str, Any] | None = None
    """A dictionary of additional attributes associated with the text unit (optional)."""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "short_id",
        text_key: str = "text",
        text_embedding_key: str = "text_embedding",
        entities_key: str = "entity_ids",
        relationships_key: str = "relationship_ids",
        covariates_key: str = "covariate_ids",
        n_tokens_key: str = "n_tokens",
        document_ids_key: str = "document_ids",
        attributes_key: str = "attributes",
    ) -> "TextUnit":
        """Create a new text unit from the dict data."""
        return TextUnit(
            id=d[id_key],
            short_id=d.get(short_id_key),
            text=d[text_key],
            text_embedding=d.get(text_embedding_key),
            entity_ids=d.get(entities_key),
            relationship_ids=d.get(relationships_key),
            covariate_ids=d.get(covariates_key),
            n_tokens=d.get(n_tokens_key),
            document_ids=d.get(document_ids_key),
            attributes=d.get(attributes_key),
        )


@dataclass
class Relationship(Identified):
    """A relationship between two entities. This is a generic relationship, and can be used to represent any type of relationship between any two entities."""

    source: str
    """The source entity name."""

    target: str
    """The target entity name."""

    weight: float | None = 1.0
    """The edge weight."""

    description: str | None = None
    """A description of the relationship (optional)."""

    description_embedding: list[float] | None = None
    """The semantic embedding for the relationship description (optional)."""

    text_unit_ids: list[str] | None = None
    """List of text unit IDs in which the relationship appears (optional)."""

    document_ids: list[str] | None = None
    """List of document IDs in which the relationship appears (optional)."""

    attributes: dict[str, Any] | None = None
    """Additional attributes associated with the relationship (optional). To be included in the search prompt"""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "short_id",
        source_key: str = "source",
        target_key: str = "target",
        description_key: str = "description",
        weight_key: str = "weight",
        text_unit_ids_key: str = "text_unit_ids",
        document_ids_key: str = "document_ids",
        attributes_key: str = "attributes",
    ) -> "Relationship":
        """Create a new relationship from the dict data."""
        return Relationship(
            id=d[id_key],
            short_id=d.get(short_id_key),
            source=d[source_key],
            target=d[target_key],
            description=d.get(description_key),
            weight=d.get(weight_key, 1.0),
            text_unit_ids=d.get(text_unit_ids_key),
            document_ids=d.get(document_ids_key),
            attributes=d.get(attributes_key),
        )


@dataclass
class CommunityReport(Named):
    """Defines an LLM-generated summary report of a community."""

    community_id: str
    """The ID of the community this report is associated with."""

    summary: str = ""
    """Summary of the report."""

    full_content: str = ""
    """Full content of the report."""

    rank: float | None = 1.0
    """Rank of the report, used for sorting (optional). Higher means more important"""

    summary_embedding: list[float] | None = None
    """The semantic (i.e. text) embedding of the report summary (optional)."""

    full_content_embedding: list[float] | None = None
    """The semantic (i.e. text) embedding of the full report content (optional)."""

    attributes: dict[str, Any] | None = None
    """A dictionary of additional attributes associated with the report (optional)."""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        title_key: str = "title",
        community_id_key: str = "community_id",
        short_id_key: str = "short_id",
        summary_key: str = "summary",
        full_content_key: str = "full_content",
        rank_key: str = "rank",
        summary_embedding_key: str = "summary_embedding",
        full_content_embedding_key: str = "full_content_embedding",
        attributes_key: str = "attributes",
    ) -> "CommunityReport":
        """Create a new community report from the dict data."""
        return CommunityReport(
            id=d[id_key],
            title=d[title_key],
            community_id=d[community_id_key],
            short_id=d.get(short_id_key),
            summary=d[summary_key],
            full_content=d[full_content_key],
            rank=d[rank_key],
            summary_embedding=d.get(summary_embedding_key),
            full_content_embedding=d.get(full_content_embedding_key),
            attributes=d.get(attributes_key),
        )


@dataclass
class Covariate(Identified):
    """
    A protocol for a covariate in the system.

    Covariates are metadata associated with a subject, e.g. entity claims.
    Each subject (e.g. entity) may be associated with multiple types of covariates.
    """

    subject_id: str
    """The subject id."""

    subject_type: str = "entity"
    """The subject type."""

    covariate_type: str = "claim"
    """The covariate type."""

    text_unit_ids: list[str] | None = None
    """List of text unit IDs in which the covariate info appears (optional)."""

    document_ids: list[str] | None = None
    """List of document IDs in which the covariate info appears (optional)."""

    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        subject_id_key: str = "subject_id",
        subject_type_key: str = "subject_type",
        covariate_type_key: str = "covariate_type",
        short_id_key: str = "short_id",
        text_unit_ids_key: str = "text_unit_ids",
        document_ids_key: str = "document_ids",
        attributes_key: str = "attributes",
    ) -> "Covariate":
        """Create a new covariate from the dict data."""
        return Covariate(
            id=d[id_key],
            short_id=d.get(short_id_key),
            subject_id=d[subject_id_key],
            subject_type=d.get(subject_type_key, "entity"),
            covariate_type=d.get(covariate_type_key, "claim"),
            text_unit_ids=d.get(text_unit_ids_key),
            document_ids=d.get(document_ids_key),
            attributes=d.get(attributes_key),
        )


@dataclass
class Community(Named):
    """A protocol for a community in the system."""

    level: str = ""
    """Community level."""

    entity_ids: list[str] | None = None
    """List of entity IDs related to the community (optional)."""

    relationship_ids: list[str] | None = None
    """List of relationship IDs related to the community (optional)."""

    covariate_ids: dict[str, list[str]] | None = None
    """Dictionary of different types of covariates related to the community (optional), e.g. claims"""

    attributes: dict[str, Any] | None = None
    """A dictionary of additional attributes associated with the community (optional). To be included in the search prompt."""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        title_key: str = "title",
        short_id_key: str = "short_id",
        level_key: str = "level",
        entities_key: str = "entity_ids",
        relationships_key: str = "relationship_ids",
        covariates_key: str = "covariate_ids",
        attributes_key: str = "attributes",
    ) -> "Community":
        """Create a new community from the dict data."""
        return Community(
            id=d[id_key],
            title=d[title_key],
            short_id=d.get(short_id_key),
            level=d[level_key],
            entity_ids=d.get(entities_key),
            relationship_ids=d.get(relationships_key),
            covariate_ids=d.get(covariates_key),
            attributes=d.get(attributes_key),
        )


@dataclass
class Entity(Named):
    """A protocol for an entity in the system."""

    type: str | None = None
    """Type of the entity (can be any string, optional)."""

    description: str | None = None
    """Description of the entity (optional)."""

    description_embedding: list[float] | None = None
    """The semantic (i.e. text) embedding of the entity (optional)."""

    name_embedding: list[float] | None = None
    """The semantic (i.e. text) embedding of the entity (optional)."""

    graph_embedding: list[float] | None = None
    """The graph embedding of the entity, likely from node2vec (optional)."""

    community_ids: list[str] | None = None
    """The community IDs of the entity (optional)."""

    text_unit_ids: list[str] | None = None
    """List of text unit IDs in which the entity appears (optional)."""

    document_ids: list[str] | None = None
    """List of document IDs in which the entity appears (optional)."""

    rank: int | None = 1
    """Rank of the entity, used for sorting (optional). Higher rank indicates more important entity. This can be based on centrality or other metrics."""

    attributes: dict[str, Any] | None = None
    """Additional attributes associated with the entity (optional), e.g. start time, end time, etc. To be included in the search prompt."""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "short_id",
        title_key: str = "title",
        type_key: str = "type",
        description_key: str = "description",
        description_embedding_key: str = "description_embedding",
        name_embedding_key: str = "name_embedding",
        graph_embedding_key: str = "graph_embedding",
        community_key: str = "community",
        text_unit_ids_key: str = "text_unit_ids",
        document_ids_key: str = "document_ids",
        rank_key: str = "degree",
        attributes_key: str = "attributes",
    ) -> "Entity":
        """Create a new entity from the dict data."""
        return Entity(
            id=d[id_key],
            title=d[title_key],
            short_id=d.get(short_id_key),
            type=d.get(type_key),
            description=d.get(description_key),
            name_embedding=d.get(name_embedding_key),
            description_embedding=d.get(description_embedding_key),
            graph_embedding=d.get(graph_embedding_key),
            community_ids=d.get(community_key),
            rank=d.get(rank_key, 1),
            text_unit_ids=d.get(text_unit_ids_key),
            document_ids=d.get(document_ids_key),
            attributes=d.get(attributes_key),
        )


def read_entities(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str | None = "short_id",
    title_col: str = "title",
    type_col: str | None = "type",
    description_col: str | None = "description",
    name_embedding_col: str | None = "name_embedding",
    description_embedding_col: str | None = "description_embedding",
    graph_embedding_col: str | None = "graph_embedding",
    community_col: str | None = "community_ids",
    text_unit_ids_col: str | None = "text_unit_ids",
    document_ids_col: str | None = "document_ids",
    rank_col: str | None = "degree",
    attributes_cols: list[str] | None = None,
) -> list[Entity]:
    """Read entities from a dataframe."""
    entities = []
    for idx, row in df.iterrows():
        entity = Entity(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            type=to_optional_str(row, type_col),
            description=to_optional_str(row, description_col),
            name_embedding=to_optional_list(row, name_embedding_col, item_type=float),
            description_embedding=to_optional_list(
                row, description_embedding_col, item_type=float
            ),
            graph_embedding=to_optional_list(row, graph_embedding_col, item_type=float),
            community_ids=to_optional_list(row, community_col, item_type=str),
            text_unit_ids=to_optional_list(row, text_unit_ids_col),
            document_ids=to_optional_list(row, document_ids_col),
            rank=to_optional_int(row, rank_col),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        entities.append(entity)
    return entities


@dataclass
class Document(Named):
    """A protocol for a document in the system."""

    type: str = "text"
    """Type of the document."""

    text_unit_ids: list[str] = field(default_factory=list)
    """list of text units in the document."""

    raw_content: str = ""
    """The raw text content of the document."""

    summary: str | None = None
    """Summary of the document (optional)."""

    summary_embedding: list[float] | None = None
    """The semantic embedding for the document summary (optional)."""

    raw_content_embedding: list[float] | None = None
    """The semantic embedding for the document raw content (optional)."""

    attributes: dict[str, Any] | None = None
    """A dictionary of structured attributes such as author, etc (optional)."""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "short_id",
        title_key: str = "title",
        type_key: str = "type",
        raw_content_key: str = "raw_content",
        summary_key: str = "summary",
        summary_embedding_key: str = "summary_embedding",
        raw_content_embedding_key: str = "raw_content_embedding",
        text_units_key: str = "text_units",
        attributes_key: str = "attributes",
    ) -> "Document":
        """Create a new document from the dict data."""
        return Document(
            id=d[id_key],
            short_id=d.get(short_id_key),
            title=d[title_key],
            type=d.get(type_key, "text"),
            raw_content=d[raw_content_key],
            summary=d.get(summary_key),
            summary_embedding=d.get(summary_embedding_key),
            raw_content_embedding=d.get(raw_content_embedding_key),
            text_unit_ids=d.get(text_units_key, []),
            attributes=d.get(attributes_key),
        )


@dataclass
class VectorStoreDocument:
    """A document that is stored in vector storage."""

    id: str | int
    """unique id for the document"""

    text: str | None
    vector: list[float] | None

    attributes: dict[str, Any] = field(default_factory=dict)
    """store any additional metadata, e.g. title, date ranges, etc"""


TextEmbedder = Callable[[str], list[float]]


@dataclass
class VectorStoreSearchResult:
    """A vector storage search result."""

    document: VectorStoreDocument
    """Document that was found."""

    score: float
    """Similarity score between 0 and 1. Higher is more similar."""


class BaseVectorStore(ABC):
    """The base class for vector storage data-access classes."""

    def __init__(
        self,
        collection_name: str,
        db_connection: Any | None = None,
        document_collection: Any | None = None,
        query_filter: Any | None = None,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self.db_connection = db_connection
        self.document_collection = document_collection
        self.query_filter = query_filter
        self.kwargs = kwargs

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Connect to vector storage."""

    @abstractmethod
    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into the vector-store."""

    @abstractmethod
    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform ANN search by vector."""

    @abstractmethod
    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform ANN search by text."""

    @abstractmethod
    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""


def store_entity_semantic_embeddings(
    entities: list[Entity],
    vectorstore: BaseVectorStore,
) -> BaseVectorStore:
    """Store entity semantic embeddings in a vectorstore."""
    documents = [
        VectorStoreDocument(
            id=entity.id,
            text=entity.description,
            vector=entity.description_embedding,
            attributes=(
                {"title": entity.title, **entity.attributes}
                if entity.attributes
                else {"title": entity.title}
            ),
        )
        for entity in entities
    ]
    vectorstore.load_documents(documents=documents)
    return vectorstore


def store_entity_behavior_embeddings(
    entities: list[Entity],
    vectorstore: BaseVectorStore,
) -> BaseVectorStore:
    """Store entity behavior embeddings in a vectorstore."""
    documents = [
        VectorStoreDocument(
            id=entity.id,
            text=entity.description,
            vector=entity.graph_embedding,
            attributes=(
                {"title": entity.title, **entity.attributes}
                if entity.attributes
                else {"title": entity.title}
            ),
        )
        for entity in entities
    ]
    vectorstore.load_documents(documents=documents)
    return vectorstore


def read_relationships(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str | None = "short_id",
    source_col: str = "source",
    target_col: str = "target",
    description_col: str | None = "description",
    description_embedding_col: str | None = "description_embedding",
    weight_col: str | None = "weight",
    text_unit_ids_col: str | None = "text_unit_ids",
    document_ids_col: str | None = "document_ids",
    attributes_cols: list[str] | None = None,
) -> list[Relationship]:
    """Read relationships from a dataframe."""
    relationships = []
    for idx, row in df.iterrows():
        rel = Relationship(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            source=to_str(row, source_col),
            target=to_str(row, target_col),
            description=to_optional_str(row, description_col),
            description_embedding=to_optional_list(
                row, description_embedding_col, item_type=float
            ),
            weight=to_optional_float(row, weight_col),
            text_unit_ids=to_optional_list(row, text_unit_ids_col, item_type=str),
            document_ids=to_optional_list(row, document_ids_col, item_type=str),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        relationships.append(rel)
    return relationships


def read_covariates(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str | None = "short_id",
    subject_col: str = "subject_id",
    subject_type_col: str | None = "subject_type",
    covariate_type_col: str | None = "covariate_type",
    text_unit_ids_col: str | None = "text_unit_ids",
    document_ids_col: str | None = "document_ids",
    attributes_cols: list[str] | None = None,
) -> list[Covariate]:
    """Read covariates from a dataframe."""
    covariates = []
    for idx, row in df.iterrows():
        cov = Covariate(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            subject_id=to_str(row, subject_col),
            subject_type=(
                to_str(row, subject_type_col) if subject_type_col else "entity"
            ),
            covariate_type=(
                to_str(row, covariate_type_col) if covariate_type_col else "claim"
            ),
            text_unit_ids=to_optional_list(row, text_unit_ids_col, item_type=str),
            document_ids=to_optional_list(row, document_ids_col, item_type=str),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        covariates.append(cov)
    return covariates


def read_communities(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str | None = "short_id",
    title_col: str = "title",
    level_col: str = "level",
    entities_col: str | None = "entity_ids",
    relationships_col: str | None = "relationship_ids",
    covariates_col: str | None = "covariate_ids",
    attributes_cols: list[str] | None = None,
) -> list[Community]:
    """Read communities from a dataframe."""
    communities = []
    for idx, row in df.iterrows():
        comm = Community(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            level=to_str(row, level_col),
            entity_ids=to_optional_list(row, entities_col, item_type=str),
            relationship_ids=to_optional_list(row, relationships_col, item_type=str),
            covariate_ids=to_optional_dict(
                row, covariates_col, key_type=str, value_type=str
            ),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        communities.append(comm)
    return communities


def read_community_reports(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str | None = "short_id",
    title_col: str = "title",
    community_col: str = "community",
    summary_col: str = "summary",
    content_col: str = "full_content",
    rank_col: str | None = "rank",
    summary_embedding_col: str | None = "summary_embedding",
    content_embedding_col: str | None = "full_content_embedding",
    attributes_cols: list[str] | None = None,
) -> list[CommunityReport]:
    """Read community reports from a dataframe."""
    reports = []
    for idx, row in df.iterrows():
        report = CommunityReport(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            community_id=to_str(row, community_col),
            summary=to_str(row, summary_col),
            full_content=to_str(row, content_col),
            rank=to_optional_float(row, rank_col),
            summary_embedding=to_optional_list(
                row, summary_embedding_col, item_type=float
            ),
            full_content_embedding=to_optional_list(
                row, content_embedding_col, item_type=float
            ),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        reports.append(report)
    return reports


def read_text_units(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str | None = "short_id",
    text_col: str = "text",
    entities_col: str | None = "entity_ids",
    relationships_col: str | None = "relationship_ids",
    covariates_col: str | None = "covariate_ids",
    tokens_col: str | None = "n_tokens",
    document_ids_col: str | None = "document_ids",
    embedding_col: str | None = "text_embedding",
    attributes_cols: list[str] | None = None,
) -> list[TextUnit]:
    """Read text units from a dataframe."""
    text_units = []
    for idx, row in df.iterrows():
        chunk = TextUnit(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            text=to_str(row, text_col),
            entity_ids=to_optional_list(row, entities_col, item_type=str),
            relationship_ids=to_optional_list(row, relationships_col, item_type=str),
            covariate_ids=to_optional_dict(
                row, covariates_col, key_type=str, value_type=str
            ),
            text_embedding=to_optional_list(row, embedding_col, item_type=float),  # type: ignore
            n_tokens=to_optional_int(row, tokens_col),
            document_ids=to_optional_list(row, document_ids_col, item_type=str),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        text_units.append(chunk)
    return text_units


def read_documents(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str = "short_id",
    title_col: str = "title",
    type_col: str = "type",
    summary_col: str | None = "entities",
    raw_content_col: str | None = "relationships",
    summary_embedding_col: str | None = "summary_embedding",
    content_embedding_col: str | None = "raw_content_embedding",
    text_units_col: str | None = "text_units",
    attributes_cols: list[str] | None = None,
) -> list[Document]:
    """Read documents from a dataframe."""
    docs = []
    for idx, row in df.iterrows():
        doc = Document(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            type=to_str(row, type_col),
            summary=to_optional_str(row, summary_col),
            raw_content=to_str(row, raw_content_col),
            summary_embedding=to_optional_list(
                row, summary_embedding_col, item_type=float
            ),
            raw_content_embedding=to_optional_list(
                row, content_embedding_col, item_type=float
            ),
            text_units=to_list(row, text_units_col, item_type=str),  # type: ignore
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        docs.append(doc)
    return docs


def read_indexer_text_units(final_text_units: pd.DataFrame) -> list[TextUnit]:
    """Read in the Text Units from the raw indexing outputs."""
    return read_text_units(
        df=final_text_units,
        short_id_col=None,
        # expects a covariate map of type -> ids
        covariates_col=None,
    )


def read_indexer_covariates(final_covariates: pd.DataFrame) -> list[Covariate]:
    """Read in the Claims from the raw indexing outputs."""
    covariate_df = final_covariates
    covariate_df["id"] = covariate_df["id"].astype(str)
    return read_covariates(
        df=covariate_df,
        short_id_col="human_readable_id",
        attributes_cols=[
            "object_id",
            "status",
            "start_date",
            "end_date",
            "description",
        ],
        text_unit_ids_col=None,
    )


def read_indexer_relationships(final_relationships: pd.DataFrame) -> list[Relationship]:
    """Read in the Relationships from the raw indexing outputs."""
    return read_relationships(
        df=final_relationships,
        short_id_col="human_readable_id",
        description_embedding_col=None,
        document_ids_col=None,
        attributes_cols=["rank"],
    )


def read_indexer_reports(
    final_community_reports: pd.DataFrame,
    final_nodes: pd.DataFrame,
    community_level: int,
) -> list[CommunityReport]:
    """Read in the Community Reports from the raw indexing outputs."""
    report_df = final_community_reports
    entity_df = final_nodes
    entity_df = _filter_under_community_level(entity_df, community_level)
    entity_df["community"] = entity_df["community"].fillna(-1)
    entity_df["community"] = entity_df["community"].astype(int)

    entity_df = entity_df.groupby(["title"]).agg({"community": "max"}).reset_index()
    entity_df["community"] = entity_df["community"].astype(str)
    filtered_community_df = entity_df["community"].drop_duplicates()

    report_df = _filter_under_community_level(report_df, community_level)
    report_df = report_df.merge(filtered_community_df, on="community", how="inner")

    return read_community_reports(
        df=report_df,
        id_col="community",
        short_id_col="community",
        summary_embedding_col=None,
        content_embedding_col=None,
    )


def read_indexer_entities(
    final_nodes: pd.DataFrame,
    final_entities: pd.DataFrame,
    community_level: int,
) -> list[Entity]:
    """Read in the Entities from the raw indexing outputs."""
    entity_df = final_nodes
    entity_embedding_df = final_entities

    entity_df = _filter_under_community_level(entity_df, community_level)
    entity_df = cast(pd.DataFrame, entity_df[["title", "degree", "community"]]).rename(
        columns={"title": "name", "degree": "rank"}
    )

    entity_df["community"] = entity_df["community"].fillna(-1)
    entity_df["community"] = entity_df["community"].astype(int)
    entity_df["rank"] = entity_df["rank"].astype(int)

    # for duplicate entities, keep the one with the highest community level
    entity_df = (
        entity_df.groupby(["name", "rank"]).agg({"community": "max"}).reset_index()
    )
    entity_df["community"] = entity_df["community"].apply(lambda x: [str(x)])
    entity_df = entity_df.merge(
        entity_embedding_df, on="name", how="inner"
    ).drop_duplicates(subset=["name"])

    # read entity dataframe to knowledge model objects
    return read_entities(
        df=entity_df,
        id_col="id",
        title_col="name",
        type_col="type",
        short_id_col="human_readable_id",
        description_col="description",
        community_col="community",
        rank_col="rank",
        name_embedding_col=None,
        description_embedding_col="description_embedding",
        graph_embedding_col=None,
        text_unit_ids_col="text_unit_ids",
        document_ids_col=None,
    )


def _filter_under_community_level(
    df: pd.DataFrame, community_level: int
) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        df[df.level <= community_level],
    )
