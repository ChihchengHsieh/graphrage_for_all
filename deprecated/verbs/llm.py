from typing import Callable, List, Dict


def execute_llm(
    input: str,
    send_to: Callable[[List[Dict[str, str]]], str],
    # extraction_prompt: str,
    # variables: Dict | None = None,
    history: List = [],
    model_args: Dict | None = {},
):
    # input = perform_variable_replacements(extraction_prompt, history, variables)
    messages = []
    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": input,
        }
    )

    response = send_to(messages, **model_args)  # modify this for other llms.
    history = [*history, {"role": "system", "content": response}]
    return response, history


def attach_history(
    input: str, history: List[Dict[str, str]] = []
) -> List[Dict[str, str]]:
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
