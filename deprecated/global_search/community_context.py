from typing import Any, cast
import tiktoken, random
import pandas as pd
from readers import Entity, CommunityReport

from dataclasses import dataclass
from enum import Enum


def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    return len(token_encoder.encode(text))  # type: ignore


class ConversationRole(str, Enum):
    """Enum for conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    @staticmethod
    def from_string(value: str) -> "ConversationRole":
        """Convert string to ConversationRole."""
        if value == "system":
            return ConversationRole.SYSTEM
        if value == "user":
            return ConversationRole.USER
        if value == "assistant":
            return ConversationRole.ASSISTANT

        msg = f"Invalid Role: {value}"
        raise ValueError(msg)

    def __str__(self) -> str:
        """Return string representation of the enum value."""
        return self.value


@dataclass
class ConversationTurn:
    """Data class for storing a single conversation turn."""

    role: ConversationRole
    content: str

    def __str__(self) -> str:
        """Return string representation of the conversation turn."""
        return f"{self.role}: {self.content}"


@dataclass
class QATurn:
    """
    Data class for storing a QA turn.

    A QA turn contains a user question and one more multiple assistant answers.
    """

    user_query: ConversationTurn
    assistant_answers: list[ConversationTurn] | None = None

    def get_answer_text(self) -> str | None:
        """Get the text of the assistant answers."""
        return (
            "\n".join([answer.content for answer in self.assistant_answers])
            if self.assistant_answers
            else None
        )

    def __str__(self) -> str:
        """Return string representation of the QA turn."""
        answers = self.get_answer_text()
        return (
            f"Question: {self.user_query.content}\nAnswer: {answers}"
            if answers
            else f"Question: {self.user_query.content}"
        )


@dataclass
class ConversationTurn:
    """Data class for storing a single conversation turn."""

    role: ConversationRole
    content: str

    def __str__(self) -> str:
        """Return string representation of the conversation turn."""
        return f"{self.role}: {self.content}"


class ConversationHistory:
    """Class for storing a conversation history."""

    turns: list[ConversationTurn]

    def __init__(self):
        self.turns = []

    @classmethod
    def from_list(
        cls, conversation_turns: list[dict[str, str]]
    ) -> "ConversationHistory":
        """
        Create a conversation history from a list of conversation turns.

        Each turn is a dictionary in the form of {"role": "<conversation_role>", "content": "<turn content>"}
        """
        history = cls()
        for turn in conversation_turns:
            history.turns.append(
                ConversationTurn(
                    role=ConversationRole.from_string(
                        turn.get("role", ConversationRole.USER)
                    ),
                    content=turn.get("content", ""),
                )
            )
        return history

    def add_turn(self, role: ConversationRole, content: str):
        """Add a new turn to the conversation history."""
        self.turns.append(ConversationTurn(role=role, content=content))

    def to_qa_turns(self) -> list[QATurn]:
        """Convert conversation history to a list of QA turns."""
        qa_turns = list[QATurn]()
        current_qa_turn = None
        for turn in self.turns:
            if turn.role == ConversationRole.USER:
                if current_qa_turn:
                    qa_turns.append(current_qa_turn)
                current_qa_turn = QATurn(user_query=turn, assistant_answers=[])
            else:
                if current_qa_turn:
                    current_qa_turn.assistant_answers.append(turn)  # type: ignore
        if current_qa_turn:
            qa_turns.append(current_qa_turn)
        return qa_turns

    def get_user_turns(self, max_user_turns: int | None = 1) -> list[str]:
        """Get the last user turns in the conversation history."""
        user_turns = []
        for turn in self.turns[::-1]:
            if turn.role == ConversationRole.USER:
                user_turns.append(turn.content)
                if max_user_turns and len(user_turns) >= max_user_turns:
                    break
        return user_turns

    def build_context(
        self,
        token_encoder: tiktoken.Encoding | None = None,
        include_user_turns_only: bool = True,
        max_qa_turns: int | None = 5,
        max_tokens: int = 8000,
        recency_bias: bool = True,
        column_delimiter: str = "|",
        context_name: str = "Conversation History",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """
        Prepare conversation history as context data for system prompt.

        Parameters
        ----------
            user_queries_only: If True, only user queries (not assistant responses) will be included in the context, default is True.
            max_qa_turns: Maximum number of QA turns to include in the context, default is 1.
            recency_bias: If True, reverse the order of the conversation history to ensure last QA got prioritized.
            column_delimiter: Delimiter to use for separating columns in the context data, default is "|".
            context_name: Name of the context, default is "Conversation History".

        """
        qa_turns = self.to_qa_turns()
        if include_user_turns_only:
            qa_turns = [
                QATurn(user_query=qa_turn.user_query, assistant_answers=None)
                for qa_turn in qa_turns
            ]
        if recency_bias:
            qa_turns = qa_turns[::-1]
        if max_qa_turns and len(qa_turns) > max_qa_turns:
            qa_turns = qa_turns[:max_qa_turns]

        # build context for qa turns
        # add context header
        if len(qa_turns) == 0 or not qa_turns:
            return ("", {context_name: pd.DataFrame()})

        # add table header
        header = f"-----{context_name}-----" + "\n"

        turn_list = []
        current_context_df = pd.DataFrame()
        for turn in qa_turns:
            turn_list.append(
                {
                    "turn": ConversationRole.USER.__str__(),
                    "content": turn.user_query.content,
                }
            )
            if turn.assistant_answers:
                turn_list.append(
                    {
                        "turn": ConversationRole.ASSISTANT.__str__(),
                        "content": turn.get_answer_text(),
                    }
                )

            context_df = pd.DataFrame(turn_list)
            context_text = header + context_df.to_csv(sep=column_delimiter, index=False)
            if num_tokens(context_text, token_encoder) > max_tokens:
                break

            current_context_df = context_df
        context_text = header + current_context_df.to_csv(
            sep=column_delimiter, index=False
        )
        return (context_text, {context_name.lower(): current_context_df})


def _rank_report_context(
    report_df: pd.DataFrame,
    weight_column: str | None = "occurrence weight",
    rank_column: str | None = "rank",
) -> pd.DataFrame:
    """Sort report context by community weight and rank if exist."""
    rank_attributes = []
    if weight_column:
        rank_attributes.append(weight_column)
        report_df[weight_column] = report_df[weight_column].astype(float)
    if rank_column:
        rank_attributes.append(rank_column)
        report_df[rank_column] = report_df[rank_column].astype(float)
    if len(rank_attributes) > 0:
        report_df.sort_values(by=rank_attributes, ascending=False, inplace=True)
    return report_df


def _convert_report_context_to_df(
    context_records: list[list[str]],
    header: list[str],
    weight_column: str | None = None,
    rank_column: str | None = None,
) -> pd.DataFrame:
    """Convert report context records to pandas dataframe and sort by weight and rank if exist."""
    record_df = pd.DataFrame(
        context_records,
        columns=cast(Any, header),
    )
    return _rank_report_context(
        report_df=record_df,
        weight_column=weight_column,
        rank_column=rank_column,
    )


def _compute_community_weights(
    community_reports: list[CommunityReport],
    entities: list[Entity],
    weight_attribute: str = "occurrence",
    normalize: bool = True,
) -> list[CommunityReport]:
    """Calculate a community's weight as count of text units associated with entities within the community."""
    community_text_units = {}
    for entity in entities:
        if entity.community_ids:
            for community_id in entity.community_ids:
                if community_id not in community_text_units:
                    community_text_units[community_id] = []
                community_text_units[community_id].extend(entity.text_unit_ids)
    for report in community_reports:
        if not report.attributes:
            report.attributes = {}
        report.attributes[weight_attribute] = len(
            set(community_text_units.get(report.community_id, []))
        )
    if normalize:
        # normalize by max weight
        all_weights = [
            report.attributes[weight_attribute]
            for report in community_reports
            if report.attributes
        ]
        max_weight = max(all_weights)
        for report in community_reports:
            if report.attributes:
                report.attributes[weight_attribute] = (
                    report.attributes[weight_attribute] / max_weight
                )
    return community_reports


def build_community_context(
    community_reports: list[CommunityReport],
    entities: list[Entity] | None = None,
    token_encoder: tiktoken.Encoding | None = None,
    use_community_summary: bool = True,
    column_delimiter: str = "|",
    shuffle_data: bool = True,
    include_community_rank: bool = False,
    min_community_rank: int = 0,
    community_rank_name: str = "rank",
    include_community_weight: bool = True,
    community_weight_name: str = "occurrence weight",
    normalize_community_weight: bool = True,
    max_tokens: int = 8000,
    single_batch: bool = True,
    context_name: str = "Reports",
    random_state: int = 86,
) -> tuple[str | list[str], dict[str, pd.DataFrame]]:
    """
    Prepare community report data table as context data for system prompt.

    If entities are provided, the community weight is calculated as the count of text units associated with entities within the community.

    The calculated weight is added as an attribute to the community reports and added to the context data table.
    """
    if (
        entities
        and len(community_reports) > 0
        and include_community_weight
        and (
            community_reports[0].attributes is None
            or community_weight_name not in community_reports[0].attributes
        )
    ):
        print("Computing community weights...")
        community_reports = _compute_community_weights(
            community_reports=community_reports,
            entities=entities,
            weight_attribute=community_weight_name,
            normalize=normalize_community_weight,
        )

    selected_reports = [
        report
        for report in community_reports
        if report.rank and report.rank >= min_community_rank
    ]
    if selected_reports is None or len(selected_reports) == 0:
        return ([], {})

    if shuffle_data:
        random.seed(random_state)
        random.shuffle(selected_reports)

    # add context header
    current_context_text = f"-----{context_name}-----" + "\n"

    # add header
    header = ["id", "title"]
    attribute_cols = (
        list(selected_reports[0].attributes.keys())
        if selected_reports[0].attributes
        else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    if not include_community_weight:
        attribute_cols = [col for col in attribute_cols if col != community_weight_name]
    header.extend(attribute_cols)
    header.append("summary" if use_community_summary else "content")
    if include_community_rank:
        header.append(community_rank_name)

    current_context_text += column_delimiter.join(header) + "\n"
    current_tokens = num_tokens(current_context_text, token_encoder)
    current_context_records = [header]
    all_context_text = []
    all_context_records = []

    for report in selected_reports:
        new_context = [
            report.short_id,
            report.title,
            *[
                str(report.attributes.get(field, "")) if report.attributes else ""
                for field in attribute_cols
            ],
        ]
        new_context.append(
            report.summary if use_community_summary else report.full_content
        )
        if include_community_rank:
            new_context.append(str(report.rank))
        new_context_text = column_delimiter.join(new_context) + "\n"

        new_tokens = num_tokens(new_context_text, token_encoder)
        if current_tokens + new_tokens > max_tokens:
            # convert the current context records to pandas dataframe and sort by weight and rank if exist
            if len(current_context_records) > 1:
                record_df = _convert_report_context_to_df(
                    context_records=current_context_records[1:],
                    header=current_context_records[0],
                    weight_column=(
                        community_weight_name
                        if entities and include_community_weight
                        else None
                    ),
                    rank_column=community_rank_name if include_community_rank else None,
                )

            else:
                record_df = pd.DataFrame()
            current_context_text = record_df.to_csv(index=False, sep=column_delimiter)

            if single_batch:
                return current_context_text, {context_name.lower(): record_df}

            all_context_text.append(current_context_text)
            all_context_records.append(record_df)

            # start a new batch
            current_context_text = (
                f"-----{context_name}-----"
                + "\n"
                + column_delimiter.join(header)
                + "\n"
            )
            current_tokens = num_tokens(current_context_text, token_encoder)
            current_context_records = [header]
        else:
            current_context_text += new_context_text
            current_tokens += new_tokens
            current_context_records.append(new_context)

    # add the last batch if it has not been added
    if current_context_text not in all_context_text:
        if len(current_context_records) > 1:
            record_df = _convert_report_context_to_df(
                context_records=current_context_records[1:],
                header=current_context_records[0],
                weight_column=(
                    community_weight_name
                    if entities and include_community_weight
                    else None
                ),
                rank_column=community_rank_name if include_community_rank else None,
            )
        else:
            record_df = pd.DataFrame()
        all_context_records.append(record_df)
        current_context_text = record_df.to_csv(index=False, sep=column_delimiter)
        all_context_text.append(current_context_text)

    return all_context_text, {
        context_name.lower(): pd.concat(all_context_records, ignore_index=True)
    }


class GlobalCommunityContext:
    """GlobalSearch community context builder."""

    def __init__(
        self,
        community_reports: list[CommunityReport],
        entities: list[Entity] | None = None,
        token_encoder: tiktoken.Encoding | None = None,
        random_state: int = 86,
    ):
        self.community_reports = community_reports
        self.entities = entities
        self.token_encoder = token_encoder
        self.random_state = random_state

    def build_context(
        self,
        conversation_history: ConversationHistory | None = None,
        use_community_summary: bool = True,
        column_delimiter: str = "|",
        shuffle_data: bool = True,
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        community_rank_name: str = "rank",
        include_community_weight: bool = True,
        community_weight_name: str = "occurrence",
        normalize_community_weight: bool = True,
        max_tokens: int = 8000,
        context_name: str = "Reports",
        conversation_history_user_turns_only: bool = True,
        conversation_history_max_turns: int | None = 5,
        **kwargs: Any,
    ) -> tuple[str | list[str], dict[str, pd.DataFrame]]:
        """Prepare batches of community report data table as context data for global search."""
        conversation_history_context = ""
        final_context_data = {}
        if conversation_history:
            # build conversation history context
            (
                conversation_history_context,
                conversation_history_context_data,
            ) = conversation_history.build_context(
                include_user_turns_only=conversation_history_user_turns_only,
                max_qa_turns=conversation_history_max_turns,
                column_delimiter=column_delimiter,
                max_tokens=max_tokens,
                recency_bias=False,
            )
            if conversation_history_context != "":
                final_context_data = conversation_history_context_data

        community_context, community_context_data = build_community_context(
            community_reports=self.community_reports,
            entities=self.entities,
            token_encoder=self.token_encoder,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            shuffle_data=shuffle_data,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            community_rank_name=community_rank_name,
            include_community_weight=include_community_weight,
            community_weight_name=community_weight_name,
            normalize_community_weight=normalize_community_weight,
            max_tokens=max_tokens,
            single_batch=False,
            context_name=context_name,
            random_state=self.random_state,
        )
        if isinstance(community_context, list):
            final_context = [
                f"{conversation_history_context}\n\n{context}"
                for context in community_context
            ]
        else:
            final_context = f"{conversation_history_context}\n\n{community_context}"

        final_context_data.update(community_context_data)
        return (final_context, final_context_data)
