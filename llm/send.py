from typing import Any, Callable, List, Dict
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    output: str
    history: List[Dict[str, str]]


Messages = List[Dict[str, str]]
ModelArgs = Dict[str, Any]

ChatLLM = Callable[[Messages, ModelArgs], LLMResponse]
EmbLLM = Callable[[List[str], ModelArgs], List[List[float]]]


@dataclass
class LLMSendToConfig:
    llm_send_to: ChatLLM
    llm_model_args: Dict = field(default_factory=lambda x: {})


def perform_variable_replacements(
    template: str, history: Messages, variables: Dict | None
) -> str:
    """Perform variable replacements on the input string and in a chat log."""
    result = template

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


def create_messages(input: str, history: Messages = []) -> Messages:
    messages = []
    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": input,
        }
    )
    return messages


def replace_and_send(
    send_to: ChatLLM,
    template: str,  # real input.
    replacing_variable: Dict | None = None,  # the input will be replaced here.
    history: Messages | None = None,
    llm_args: Dict | None = None,
):
    history = history or []

    if replacing_variable is not None:
        input_message = perform_variable_replacements(
            template=template,
            history=history,
            variables=replacing_variable or {},
        )
    messages = create_messages(input_message, history)
    response = send_to(messages, llm_args or {})
    return response
