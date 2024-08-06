import traceback
from dataclasses import dataclass
from typing import Any, Callable
from .template import COMMUNITY_REPORT_PROMPT
from .llm import execute_llm


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: str
    structured_output: dict


def dict_has_keys_with_types(
    data: dict, expected_fields: list[tuple[str, type]]
) -> bool:
    """Return True if the given dictionary has the given keys with the given types."""
    for field, field_type in expected_fields:
        if field not in data:
            return False

        value = data[field]
        if not isinstance(value, field_type):
            return False
    return True


from .graph_extractor import perform_variable_replacements
import json


def try_parse_json_object(input: str) -> dict:
    """Generate JSON-string output using best-attempt prompting & parsing techniques."""
    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        print("error loading json, json=%s", input)
        raise
    else:
        if not isinstance(result, dict):
            raise TypeError
        return result


def is_response_valid(x):
    return dict_has_keys_with_types(
        x,
        [
            ("title", str),
            ("summary", str),
            ("findings", list),
            ("rating", float),
            ("rating_explanation", str),
        ],
    )


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _send_to: Callable
    _input_text_key: str
    _extraction_prompt: str
    _output_formatter_prompt: str
    # _on_error: ErrorHandlerFn
    _max_report_length: int

    def __init__(
        self,
        send_to: Callable,
        input_text_key: str | None = None,
        extraction_prompt: str | None = None,
        # on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,
    ):
        """Init method definition."""
        self._send_to = send_to
        self._input_text_key = input_text_key or "input_text"
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        # self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500

    def __call__(self, inputs: dict[str, Any]):
        """Call method definition."""
        output = None
        max_retries = 10
        attempts = 0

        response_is_valid = False
        try:

            while attempts < max_retries and (not response_is_valid):
                response, history = execute_llm(
                    input=perform_variable_replacements(
                        self._extraction_prompt,
                        [],
                        variables={self._input_text_key: inputs[self._input_text_key]},
                    ),
                    send_to=self._send_to,
                )

                # check if the response is valid
                json_response = try_parse_json_object(response)
                response_is_valid = is_response_valid(json_response)
                attempts += 1

            if not response_is_valid:
                raise TimeoutError(
                    f"Have attempted {attempts} time on community report extraction."
                )

            output = json_response
        except Exception as e:
            print("error generating community report")
            self._on_error(e, traceback.format_exc(), None)
            output = {}

        text_output = self._get_text_output(output)
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
        )

    def _get_text_output(self, parsed_output: dict) -> str:
        title = parsed_output.get("title", "Report")
        summary = parsed_output.get("summary", "")
        findings = parsed_output.get("findings", [])

        def finding_summary(finding: dict):
            if isinstance(finding, str):
                return finding
            return finding.get("summary")

        def finding_explanation(finding: dict):
            if isinstance(finding, str):
                return ""
            return finding.get("explanation")

        report_sections = "\n\n".join(
            f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
        )
        return f"# {title}\n\n{summary}\n\n{report_sections}"
